#!/usr/bin/env python3
"""Generate sparse hierarchical notes rollups and the overall README."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from notes_archive_naming import DAILY_META_RE, MONTHLY_META_RE, YEARLY_META_RE
from notes_rollup import strip_sources_footer, with_sources_footer
from notes_summary_filters import resolve_forbid_patterns, run_filtered_summary_command
from summary_model import (
    DEFAULT_CODEX_REASONING,
    DEFAULT_PROVIDER,
    PROVIDERS,
    custom_command_provenance,
    resolve_spec,
    wrapper_command,
)
from update_daily_meta_summary import (
    DEFAULT_COMMAND,
    deno_fmt,
    git_has_staged_changes,
    sha256_text,
    source_paths_for_day,
)

DEFAULT_MANIFEST = Path("scripts/notes-overall-meta-manifest.json")
DEFAULT_NOTE = Path("notes/README.md")
RESERVED_NOTE_NAMES = {"AGENTS.md", "README.md"}


@dataclass(frozen=True)
class RollupSource:
    path: Path
    key: str
    level: str
    text: str
    blob_id: str


@dataclass(frozen=True)
class RollupPlan:
    path: Path
    key: str
    level: str
    mode: str
    sources: list[RollupSource]


def git_root() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )


def run_git(args: list[str], root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def worktree_blob_id(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix()
    return run_git(["hash-object", "--", rel], root)


def load_source(path: Path, key: str, level: str, root: Path) -> RollupSource:
    return RollupSource(
        path=path,
        key=key,
        level=level,
        text=path.read_text(encoding="utf-8"),
        blob_id=worktree_blob_id(path, root),
    )


def source_days(notes_dir: Path) -> list[str]:
    days: set[str] = set()
    for path in notes_dir.glob("*.md"):
        if not path.is_file() or path.name in RESERVED_NOTE_NAMES:
            continue
        if DAILY_META_RE.match(path.name):
            continue
        match = re.match(r"^(\d{8})", path.name)
        if match:
            days.add(match.group(1))
    return sorted(days)


def representative_for_day(notes_dir: Path, day: str, root: Path) -> RollupSource | None:
    source_paths = source_paths_for_day(notes_dir, day)
    if not source_paths:
        return None
    daily_path = notes_dir / f"{day}.md"
    if len(source_paths) == 1:
        return load_source(source_paths[0], day, "note", root)
    if not daily_path.exists():
        raise SystemExit(f"missing daily summary for {day}; run update_daily_meta_summary.py first")
    return load_source(daily_path, day, "day", root)


def group_by_prefix(sources: list[RollupSource], length: int) -> dict[str, list[RollupSource]]:
    groups: dict[str, list[RollupSource]] = {}
    for source in sources:
        groups.setdefault(source.key[:length], []).append(source)
    return {key: sorted(value, key=lambda source: source.key) for key, value in sorted(groups.items())}


def source_entries(sources: list[RollupSource], root: Path) -> list[dict[str, str]]:
    return [
        {
            "note": source.path.relative_to(root).as_posix(),
            "level": source.level,
            "key": source.key,
            "blob_id": source.blob_id,
        }
        for source in sorted(sources, key=lambda item: (item.key, item.path.name))
    ]


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {
            "version": 2,
            "description": "Sparse hierarchical notes rollup manifest. Rollups are generated only when they combine multiple immediate sources; README is always generated or promoted.",
            "rollups": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") == 1:
        return {
            "version": 2,
            "description": "Sparse hierarchical notes rollup manifest. Rollups are generated only when they combine multiple immediate sources; README is always generated or promoted.",
            "rollups": {},
        }
    if data.get("version") != 2 or not isinstance(data.get("rollups"), dict):
        raise SystemExit(f"unsupported overall meta manifest format: {path}")
    return data


def write_manifest(path: Path, data: dict) -> bool:
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True


def entry_for(
    plan: RollupPlan,
    text: str,
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> dict:
    entry = {
        "note": plan.path.relative_to(root).as_posix(),
        "level": plan.level,
        "key": plan.key,
        "mode": plan.mode,
        "source_count": len(plan.sources),
        "sources": source_entries(plan.sources, root),
        "summary_hash": sha256_text(text),
    }
    if plan.mode == "promote":
        entry["summarizer"] = {"provider": "promote"}
    elif summarizer is not None:
        entry["summarizer"] = summarizer
    return entry


def expected_manifest_entry(
    plan: RollupPlan,
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> dict | None:
    if not plan.path.exists():
        return None
    return entry_for(plan, plan.path.read_text(encoding="utf-8"), root, summarizer)


def is_stale(
    plan: RollupPlan,
    manifest: dict,
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> bool:
    entry = manifest.get("rollups", {}).get(plan.path.relative_to(root).as_posix())
    expected = expected_manifest_entry(plan, root, summarizer)
    if expected is None or not entry:
        return True
    return entry != expected


def period_label(level: str, key: str) -> str:
    if level == "month":
        return f"{key[:4]}-{key[4:6]} UTC"
    if level == "year":
        return f"{key} UTC"
    return "the full archive"


def title_for(level: str, key: str) -> str:
    if level == "month":
        return f"# ValueGraft Notes: {key[:4]}-{key[4:6]}"
    if level == "year":
        return f"# ValueGraft Notes: {key}"
    return "# ValueGraft Notes Overview"


def build_prompt(plan: RollupPlan, root: Path) -> str:
    source_label = {
        "month": "day representatives",
        "year": "month representatives",
        "readme": "top-level representatives",
    }.get(plan.level, "source representatives")
    parts: list[str] = [
        f"""You are writing {plan.path.relative_to(root).as_posix()}, a {plan.level} rollup for the ValueGraft research notes archive.

Input: the complete immediate {source_label}, in chronological order. These sources may be generated summaries or, when a lower-level bucket had only one file, the single lower-level note promoted directly into this layer.

Goal: help a future project agent understand the process and history covered by {period_label(plan.level, plan.key)}. Synthesize across the immediate sources. Preserve important pivots, methodology changes, empirical results, failures/corrections, terminology decisions, and remaining open questions. Conclude with a compact current-state / handoff section for this scope.

Output: the standalone Markdown body. The caller will write your response to {plan.path.relative_to(root).as_posix()}. Return only the Markdown document content: no preamble, no code fence, no tool-call syntax, no file-writing description, and no closing status note.

Preferred shape:

- a top-level `{title_for(plan.level, plan.key)}` heading;
- one italicized opening paragraph summarizing the arc;
- a concise `**Participants/contributors:** ...` paragraph naming the users,
  assistant model identifiers, authors, and labeled subagents represented in
  the immediate sources; use only source-supported identities and never guess;
- a small number of thematic sections, not one section per input file;
- a concise current-state / handoff section at the end;
- bullets only where they make dense facts easier to scan.

Style: precise, readable prose. Avoid personal/emotional characterization, dramatic process labels, and publishing-platform logistics. Do not directly name sensitive-topic material; describe it generically as an interpretability tangent or notes-hygiene issue if needed. Do not include a `Sources` section; the caller appends a standardized linked source list after your output.

Important: you already have all source text below. Do not inspect files, do not announce an intention to inspect files, and do not emit tool-call JSON or tool-call-like syntax. Your first non-whitespace character must be `#`.

# Chronological Immediate Sources
""",
    ]
    for source in sorted(plan.sources, key=lambda item: (item.key, item.path.name)):
        rel = source.path.relative_to(root).as_posix()
        parts.append(
            f"""

## Source: {rel}

Source level: {source.level}; key={source.key}; chars={len(source.text)}.

{source.text.rstrip()}
"""
        )
    return "".join(parts)


def promote_source(plan: RollupPlan) -> str:
    if len(plan.sources) != 1:
        raise ValueError("promotion plans require exactly one source")
    return with_sources_footer(
        strip_sources_footer(plan.sources[0].text),
        plan.path,
        [plan.sources[0].path],
    )


def invalid_rollup_reason(body: str) -> str | None:
    stripped = body.lstrip()
    if not stripped.startswith("#"):
        return "output must begin with a top-level Markdown heading"
    first_chunk = stripped[:600]
    if re.search(r"(?i)\b(i'll|i will|i’m going to|i am going to|let me)\s+(check|inspect|read|look)", first_chunk):
        return "output appears to announce file inspection instead of writing the summary"
    if re.search(r'(?m)^\s*\{[^}]*"(?:path|pattern|output_mode)"', body):
        return "output appears to contain tool-call JSON"
    return None


def retry_prompt_for_invalid_rollup(prompt: str, reason: str) -> str:
    return f"""\
The previous rollup output was invalid: {reason}.

Rewrite the rollup from scratch using only the source text already included in
the original task. Do not inspect files, do not announce an intention to inspect
files, and do not emit tool-call JSON. Start immediately with the requested
top-level Markdown heading.

Original task:

{prompt}
"""


def generate_rollup(
    plan: RollupPlan,
    root: Path,
    command: str,
    forbid_patterns: list[str],
    max_forbid_attempts: int,
) -> str:
    if plan.mode == "promote":
        return promote_source(plan)
    prompt = build_prompt(plan, root)
    print(f"prompt_chars[{plan.path.relative_to(root)}]={len(prompt)}")
    current_prompt = prompt
    attempts = max(1, max_forbid_attempts)
    for attempt in range(1, attempts + 1):
        body = run_filtered_summary_command(
            command,
            current_prompt,
            forbid_patterns,
            max_forbid_attempts,
            f"{plan.level} rollup {plan.key}",
        )
        reason = invalid_rollup_reason(body)
        if reason is None:
            return with_sources_footer(body, plan.path, [source.path for source in plan.sources])
        print(
            f"invalid {plan.level} rollup {plan.key} output in attempt "
            f"{attempt}/{attempts}: {reason}"
        )
        if attempt == attempts:
            raise RuntimeError(f"{plan.level} rollup {plan.key} failed validation: {reason}")
        current_prompt = retry_prompt_for_invalid_rollup(prompt, reason)
    raise AssertionError("unreachable rollup validation retry state")


def generated_rollup_paths(notes_dir: Path) -> set[Path]:
    return {
        path
        for path in notes_dir.glob("*.md")
        if MONTHLY_META_RE.match(path.name) or YEARLY_META_RE.match(path.name) or path.name == "README.md"
    }


def build_plans(notes_dir: Path, root: Path, readme_path: Path) -> tuple[list[RollupPlan], set[Path]]:
    day_reps = [
        rep
        for day in source_days(notes_dir)
        if (rep := representative_for_day(notes_dir, day, root)) is not None
    ]
    if not day_reps:
        raise SystemExit("no note representatives found")

    plans: list[RollupPlan] = []
    month_reps: list[RollupSource] = []
    for month, sources in group_by_prefix(day_reps, 6).items():
        path = notes_dir / f"{month}.md"
        if len(sources) == 1:
            month_reps.append(sources[0])
        else:
            plan = RollupPlan(path=path, key=month, level="month", mode="summary", sources=sources)
            plans.append(plan)
            if path.exists():
                month_reps.append(load_source(path, month, "month", root))
            else:
                month_reps.append(RollupSource(path, month, "month", "", ""))

    year_reps: list[RollupSource] = []
    for year, sources in group_by_prefix(month_reps, 4).items():
        path = notes_dir / f"{year}.md"
        if len(sources) == 1:
            year_reps.append(sources[0])
        else:
            plan = RollupPlan(path=path, key=year, level="year", mode="summary", sources=sources)
            plans.append(plan)
            if path.exists():
                year_reps.append(load_source(path, year, "year", root))
            else:
                year_reps.append(RollupSource(path, year, "year", "", ""))

    readme_mode = "promote" if len(year_reps) == 1 else "summary"
    plans.append(
        RollupPlan(
            path=readme_path,
            key="overall",
            level="readme",
            mode=readme_mode,
            sources=year_reps,
        )
    )
    expected = {plan.path for plan in plans}
    return plans, expected


def refresh_plan_sources(plans: list[RollupPlan], root: Path) -> list[RollupPlan]:
    refreshed: list[RollupPlan] = []
    replacements: dict[Path, RollupSource] = {}
    for plan in plans:
        sources = [replacements.get(source.path, source) for source in plan.sources]
        refreshed_plan = RollupPlan(
            path=plan.path,
            key=plan.key,
            level=plan.level,
            mode=plan.mode,
            sources=sources,
        )
        refreshed.append(refreshed_plan)
        if plan.path.exists():
            replacements[plan.path] = load_source(plan.path, plan.key, plan.level, root)
    return refreshed


def remove_obsolete_rollups(notes_dir: Path, expected: set[Path], root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in sorted(generated_rollup_paths(notes_dir) - expected):
        rel = path.relative_to(root).as_posix()
        subprocess.check_call(["git", "rm", "--ignore-unmatch", "--", rel], cwd=root)
        if path.exists():
            path.unlink()
        removed.append(path)
    return removed


def commit_paths(root: Path, paths: list[Path], message: str) -> None:
    rels: list[str] = []
    seen: set[str] = set()
    for path in paths:
        rel = path.relative_to(root).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        rels.append(rel)
    if not rels:
        return
    subprocess.check_call(["git", "add", "-A", "--", *rels], cwd=root)
    if git_has_staged_changes(root, paths):
        subprocess.check_call(["git", "commit", "-m", message, "--", *rels], cwd=root)


def print_plan(
    plans: list[RollupPlan],
    expected: set[Path],
    manifest: dict,
    root: Path,
    summarizer: dict[str, str] | None = None,
) -> None:
    print(f"hierarchical notes rollups: plans={len(plans)} expected_generated={len(expected)}")
    for plan in plans:
        stale = is_stale(plan, manifest, root, summarizer)
        rel = plan.path.relative_to(root).as_posix()
        print(
            f"  {rel}: level={plan.level} key={plan.key} mode={plan.mode} "
            f"sources={len(plan.sources)} stale={stale}"
        )
        for source in sorted(plan.sources, key=lambda item: (item.key, item.path.name)):
            print(
                f"    - {source.path.relative_to(root).as_posix()} "
                f"level={source.level} key={source.key} blob={source.blob_id[:12]}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes-dir", type=Path, default=Path("notes"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--command")
    parser.add_argument("--summary-provider", choices=PROVIDERS, default=DEFAULT_PROVIDER)
    parser.add_argument("--summary-model")
    parser.add_argument("--summary-reasoning", default=DEFAULT_CODEX_REASONING)
    parser.add_argument("--prompt-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-command", action="store_true")
    parser.add_argument("--no-commit", action="store_true")
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
    readme_path = (root / args.note).resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = load_manifest(manifest_path)
    plans, expected = build_plans(notes_dir, root, readme_path)
    print_plan(plans, expected, manifest, root, summarizer_provenance)

    if args.prompt_out:
        args.prompt_out.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_out.write_text(
            "\n\n".join(build_prompt(plan, root) for plan in plans if plan.mode == "summary"),
            encoding="utf-8",
        )
        print(f"wrote prompt: {args.prompt_out}")

    if args.dry_run:
        return 0
    if args.no_command:
        if not args.prompt_out:
            for plan in plans:
                if plan.mode == "summary":
                    sys.stdout.write(build_prompt(plan, root))
        return 0

    forbid_patterns = resolve_forbid_patterns(
        args.forbid_regex,
        root,
        include_defaults=not args.no_default_forbid_regex,
        dotenv_paths=args.dotenv,
    )

    changed_paths: list[Path] = []
    index = 0
    while index < len(plans):
        plan = plans[index]
        stale = args.force or is_stale(plan, manifest, root, summarizer_provenance)
        if stale:
            text = generate_rollup(
                plan,
                root,
                args.command,
                forbid_patterns,
                args.max_forbid_attempts,
            )
            plan.path.parent.mkdir(parents=True, exist_ok=True)
            plan.path.write_text(text, encoding="utf-8")
            deno_fmt(root, [plan.path])
            changed_paths.append(plan.path)
        plans = refresh_plan_sources(plans, root)
        plan = plans[index]
        if plan.path.exists():
            manifest.setdefault("rollups", {})[plan.path.relative_to(root).as_posix()] = entry_for(
                plan,
                plan.path.read_text(encoding="utf-8"),
                root,
                summarizer_provenance,
            )
        index += 1

    removed = remove_obsolete_rollups(notes_dir, expected, root)
    changed_paths.extend(removed)
    expected_rels = {path.relative_to(root).as_posix() for path in expected}
    manifest["rollups"] = {
        rel: entry
        for rel, entry in sorted(manifest.get("rollups", {}).items())
        if rel in expected_rels
    }
    manifest_changed = write_manifest(manifest_path, manifest)
    if manifest_changed:
        deno_fmt(root, [manifest_path])
        changed_paths.append(manifest_path)

    if changed_paths and not args.no_commit:
        commit_paths(root, changed_paths, "Update hierarchical notes rollups")
    if not changed_paths:
        print("hierarchical notes rollups are current; no changes written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
