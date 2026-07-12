#!/usr/bin/env python3
"""Create v13 literal compact-pool permutations from a dedicated seed commit.

The seed manifest must be the sole path changed by the clean trunk HEAD.  This
command enumerates compact tuples and IDs only; it never materializes history
text.  Commit its output separately before any ranked candidate is expanded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from powered_v13_permutation import (  # noqa: E402
    LITERAL_PERMUTATION_RELATIVE_PATH,
    SEED_MANIFEST_RELATIVE_PATH,
    V13PermutationError,
    build_literal_permutations,
    parse_json_mapping_bytes,
    sha256_bytes,
    sha256_file,
    validate_construction_rng_replay,
    validate_literal_permutations,
    validate_seed_manifest,
    write_json_exclusive,
)


class PermutationWriterError(RuntimeError):
    """The committed-seed boundary is not exact."""


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


def _verify_dedicated_seed_head() -> tuple[bytes, dict, str]:
    if _git("branch", "--show-current") != "trunk":
        raise PermutationWriterError(
            "literal permutations must be created on trunk")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise PermutationWriterError(
            "literal permutations require a completely clean tree")
    relative = SEED_MANIFEST_RELATIVE_PATH
    absolute = REPO / relative
    _git("ls-files", "--error-unmatch", relative)
    head = _git("rev-parse", "HEAD")
    if _git("rev-parse", "@{upstream}") != head:
        raise PermutationWriterError("dedicated seed HEAD is not pushed")
    changed = set(filter(None, _git(
        "diff-tree", "--no-commit-id", "--name-only", "-r", head).splitlines()))
    if changed != {relative}:
        raise PermutationWriterError(
            "HEAD must be a dedicated seed-manifest-only commit")
    seed_commit = _git("log", "-1", "--format=%H", "--", relative)
    if seed_commit != head:
        raise PermutationWriterError(
            "seed manifest was not introduced by the current HEAD")
    seed_raw = _git_bytes("show", f"{head}:{relative}")
    if absolute.read_bytes() != seed_raw:
        raise PermutationWriterError(
            "worktree seed bytes differ from the committed git object")
    seed_manifest = parse_json_mapping_bytes(seed_raw, "seed manifest git object")
    validate_seed_manifest(seed_manifest)
    parent = _git("rev-parse", f"{head}^")
    boundary = seed_manifest["repository_boundary"]
    if boundary["preseed_git_commit"] != parent:
        raise PermutationWriterError(
            "dedicated seed commit parent differs from frozen pre-seed root")
    for record in seed_manifest["input_inventory"]:
        raw = _git_bytes("show", f"{parent}:{record['path']}")
        if sha256_bytes(raw) != record["sha256"]:
            raise PermutationWriterError(
                f"pre-seed git-object hash differs: {record['path']}")
    return seed_raw, seed_manifest, head


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        seed_raw, seed_manifest, seed_commit = _verify_dedicated_seed_head()
        seed_relative = SEED_MANIFEST_RELATIVE_PATH
        seed_sha = sha256_bytes(seed_raw)
        observed = datetime.now(timezone.utc).replace(
            microsecond=0).isoformat().replace("+00:00", "Z")
        permutations = build_literal_permutations(
            seed_raw,
            seed_git_commit=seed_commit,
            observed_utc=observed,
        )
        output = REPO / LITERAL_PERMUTATION_RELATIVE_PATH
        write_json_exclusive(output, permutations)
        persisted = parse_json_mapping_bytes(
            output.read_bytes(), "literal permutation output")
        validate_literal_permutations(
            persisted,
            seed_raw,
            expected_seed_git_commit=seed_commit,
        )
        validate_construction_rng_replay(
            persisted, seed_raw, expected_seed_git_commit=seed_commit)
    except (PermutationWriterError, V13PermutationError,
            subprocess.CalledProcessError, FileExistsError,
            json.JSONDecodeError, OSError) as exc:
        parser().error(str(exc))
    print(json.dumps({
        "output": str(output),
        "sha256": sha256_file(output),
        "seed_manifest_path": seed_relative,
        "seed_manifest_sha256": seed_sha,
        "seed_git_commit": seed_commit,
        "next_required_action": "DEDICATED_LITERAL_PERMUTATION_COMMIT_ONLY",
        "ranked_conversation_text_materialized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
