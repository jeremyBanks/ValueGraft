#!/usr/bin/env python3
"""Create the v13 OS-random seed manifest at a clean trunk HEAD.

This is deliberately the only operation performed by the command.  It never
constructs a permutation and never materializes an in-pool conversation.
Commit the output as a dedicated commit before running the permutation writer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from powered_v13_permutation import (  # noqa: E402
    PRESEED_INPUT_PATHS,
    SEED_MANIFEST_RELATIVE_PATH,
    SeedManifestInput,
    V13PermutationError,
    build_seed_manifest,
    parse_json_mapping_bytes,
    sha256_file,
    validate_seed_manifest,
    write_json_exclusive,
)
from powered_v13_recipe import STRATA  # noqa: E402


class SeedWriterError(RuntimeError):
    """The pre-seed repository boundary is not exact."""


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=REPO, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout


def _require_clean_integration_head() -> tuple[str, list[dict[str, str]]]:
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    if root != REPO.resolve():
        raise SeedWriterError("script repository root differs from git root")
    if _git("branch", "--show-current") != "trunk":
        raise SeedWriterError("seed freeze must be created on integration trunk")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise SeedWriterError("seed manifest requires a completely clean tree")
    head = _git("rev-parse", "HEAD")
    upstream = _git("rev-parse", "@{upstream}")
    if upstream != head:
        raise SeedWriterError("pre-seed integration HEAD is not pushed")
    inventory: list[dict[str, str]] = []
    for relative in PRESEED_INPUT_PATHS:
        _git("ls-files", "--error-unmatch", relative)
        worktree = (REPO / relative).read_bytes()
        committed = _git_bytes("show", f"{head}:{relative}")
        if worktree != committed:
            raise SeedWriterError(f"pre-seed input differs from HEAD: {relative}")
        inventory.append({
            "path": relative,
            "sha256": hashlib.sha256(committed).hexdigest(),
        })
    return head, inventory


def _os_random_seeds() -> dict[str, str]:
    seeds: dict[str, str] = {}
    for stratum in STRATA:
        while True:
            candidate = secrets.token_bytes(16).hex()
            if candidate not in seeds.values():
                seeds[stratum] = candidate
                break
    return seeds


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        head, inventory = _require_clean_integration_head()
        observed = datetime.now(timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z")
        manifest = build_seed_manifest(SeedManifestInput(
            observed_utc=observed,
            seeds=_os_random_seeds(),
            preseed_git_commit=head,
            input_inventory=inventory,
            git_clean_before_write=True,
        ))
        output = REPO / SEED_MANIFEST_RELATIVE_PATH
        write_json_exclusive(output, manifest)
        persisted = parse_json_mapping_bytes(
            output.read_bytes(), "seed manifest output")
        validate_seed_manifest(persisted)
    except (SeedWriterError, V13PermutationError, subprocess.CalledProcessError,
            FileExistsError, json.JSONDecodeError, OSError) as exc:
        parser().error(str(exc))
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "recipe_git_commit": head,
        "next_required_action": "DEDICATED_SEED_FILE_COMMIT_AND_PUSH_ONLY",
        "permutation_created": False,
        "ranked_conversation_text_materialized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
