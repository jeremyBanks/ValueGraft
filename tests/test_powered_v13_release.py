from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from powered_v13_release import (
    STAGE_A_MANIFEST_SCHEMA,
    ReleaseVerificationError,
    build_stage_a_manifest,
    canonical_json_bytes,
    create_stage_a_launch_receipt,
    encode_stage_a_manifest,
    receipt_payload_sha256,
    verify_stage_a_authorization_commit,
    verify_stage_a_checkout,
    verify_stage_a_manifest_document,
)


PREREGISTRATION = "PROTOCOL.md"
MANIFEST_PATH = "release/powered-v13-stage-a-manifest.json"
EXPERIMENT_PATHS = (PREREGISTRATION, "locks/exact.lock", "src/runner.py")

# These deliberately synthetic lines document the generic contract.  The
# release implementation contains neither the real v13 DRAFT line nor the real
# authorization line; callers must freeze both literal LF-terminated bytes.
OLD_STATUS = b"**Status:** SYNTHETIC DRAFT\n"
NEW_STATUS = b"**Status:** SYNTHETIC STATIC AUTHORIZATION\n"
T0 = datetime(2026, 7, 12, 18, 0, 0, tzinfo=timezone.utc)


def git(repo: Path, *args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr}"
        )
    return result.stdout.strip()


def write(repo: Path, relative: str, data: bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def init_static_root(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "trunk")
    git(repo, "config", "user.email", "release-test@example.com")
    git(repo, "config", "user.name", "Release Test")
    write(
        repo,
        PREREGISTRATION,
        b"# Synthetic protocol\n\n" + OLD_STATUS + b"\nNo paid work.\n",
    )
    write(repo, "src/runner.py", b"def run():\n    return 'blind'\n")
    write(repo, "locks/exact.lock", b"package==1.2.3 --hash=sha256:abcd\n")
    git(repo, "add", *EXPERIMENT_PATHS)
    git(repo, "commit", "-m", "static root")
    return repo, git(repo, "rev-parse", "HEAD")


def manifest_for(repo: Path, root: str) -> dict:
    return build_stage_a_manifest(
        repo,
        static_root_commit=root,
        experiment_paths=EXPERIMENT_PATHS,
        preregistration_path=PREREGISTRATION,
        manifest_path=MANIFEST_PATH,
        old_status_line=OLD_STATUS,
        new_status_line=NEW_STATUS,
    )


def commit_authorization(
    repo: Path,
    manifest: dict,
    *,
    preregistration_bytes: bytes | None = None,
    manifest_bytes: bytes | None = None,
    extras: dict[str, bytes] | None = None,
) -> str:
    prior = (repo / PREREGISTRATION).read_bytes()
    write(
        repo,
        PREREGISTRATION,
        preregistration_bytes
        if preregistration_bytes is not None
        else prior.replace(OLD_STATUS, NEW_STATUS, 1),
    )
    write(
        repo,
        MANIFEST_PATH,
        manifest_bytes if manifest_bytes is not None else encode_stage_a_manifest(manifest),
    )
    paths = [PREREGISTRATION, MANIFEST_PATH]
    for path, data in (extras or {}).items():
        write(repo, path, data)
        paths.append(path)
    git(repo, "add", *paths)
    git(repo, "commit", "-m", "synthetic Stage-A authorization")
    return git(repo, "rev-parse", "HEAD")


@dataclass(frozen=True)
class ValidRelease:
    repo: Path
    root: str
    authorization: str
    receipt: Path
    manifest: dict


def valid_release(tmp_path: Path, *, detach: bool = True) -> ValidRelease:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    authorization = commit_authorization(repo, manifest)
    receipt = tmp_path / "receipts" / "stage-a-launch.json"
    create_stage_a_launch_receipt(
        repo,
        receipt,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        created_at=T0,
    )
    if detach:
        git(repo, "checkout", "--detach", authorization)
    return ValidRelease(repo, root, authorization, receipt, manifest)


def test_manifest_is_canonical_explicit_parent_tree_inventory(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    binding = verify_stage_a_manifest_document(
        repo, manifest, expected_manifest_path=MANIFEST_PATH
    )

    assert manifest["schema"] == STAGE_A_MANIFEST_SCHEMA
    assert [row["path"] for row in manifest["inventory"]] == sorted(
        EXPERIMENT_PATHS
    )
    assert binding["static_root_commit"] == root
    assert encode_stage_a_manifest(manifest) == canonical_json_bytes(manifest) + b"\n"

    # A dirty live byte cannot influence a manifest over the immutable parent.
    write(repo, "src/runner.py", b"uncommitted replacement\n")
    assert manifest_for(repo, root) == manifest


def test_manifest_requires_strict_configured_one_line_transition(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    with pytest.raises(ReleaseVerificationError, match="LF-terminated"):
        build_stage_a_manifest(
            repo,
            static_root_commit=root,
            experiment_paths=EXPERIMENT_PATHS,
            preregistration_path=PREREGISTRATION,
            manifest_path=MANIFEST_PATH,
            old_status_line=OLD_STATUS,
            new_status_line=b"not-terminated",
        )
    with pytest.raises(ReleaseVerificationError, match="old status line is not unique"):
        build_stage_a_manifest(
            repo,
            static_root_commit=root,
            experiment_paths=EXPERIMENT_PATHS,
            preregistration_path=PREREGISTRATION,
            manifest_path=MANIFEST_PATH,
            old_status_line=b"**Status:** ANOTHER DRAFT\n",
            new_status_line=NEW_STATUS,
        )


def test_detached_clean_exact_release_and_fresh_receipt_pass(tmp_path: Path) -> None:
    release = valid_release(tmp_path)
    verified = verify_stage_a_checkout(
        release.repo,
        authorization_commit=release.authorization,
        manifest_path=MANIFEST_PATH,
        receipt_paths=[release.receipt],
        now=T0 + timedelta(seconds=12),
    )
    assert verified["status"] == "PASS"
    assert verified["detached_head"] == release.authorization
    assert verified["authorization"]["static_root_commit"] == release.root
    assert verified["authorization"]["changed_paths"] == [
        PREREGISTRATION,
        MANIFEST_PATH,
    ]
    assert verified["receipt"]["age_seconds"] == 12


def test_branch_checkout_is_rejected_even_at_exact_head(tmp_path: Path) -> None:
    release = valid_release(tmp_path, detach=False)
    with pytest.raises(ReleaseVerificationError, match="on branch"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt],
            now=T0,
        )


@pytest.mark.parametrize("dirty_path", ["src/runner.py", "untracked.txt"])
def test_dirty_checkout_is_rejected(tmp_path: Path, dirty_path: str) -> None:
    release = valid_release(tmp_path)
    write(release.repo, dirty_path, b"dirty\n")
    with pytest.raises(ReleaseVerificationError, match="checkout is dirty"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt],
            now=T0,
        )


def test_wrong_immediate_parent_is_rejected(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    write(repo, "orientation.txt", b"intervening commit\n")
    git(repo, "add", "orientation.txt")
    git(repo, "commit", "-m", "intervening parent")
    authorization = commit_authorization(repo, manifest)

    with pytest.raises(ReleaseVerificationError, match="parent differs"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_extra_diff_path_is_rejected(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    authorization = commit_authorization(
        repo, manifest, extras={"src/surprise.py": b"treatment = True\n"}
    )
    with pytest.raises(ReleaseVerificationError, match="diff paths/statuses differ"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_duplicate_manifest_in_authorization_diff_is_rejected(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    authorization = commit_authorization(
        repo,
        manifest,
        extras={"release/duplicate.json": encode_stage_a_manifest(manifest)},
    )
    with pytest.raises(ReleaseVerificationError, match="duplicate Stage-A manifests"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_duplicate_receipt_is_rejected_at_create_and_launch(tmp_path: Path) -> None:
    release = valid_release(tmp_path)
    with pytest.raises(ReleaseVerificationError, match="already exists"):
        create_stage_a_launch_receipt(
            release.repo,
            release.receipt,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            created_at=T0,
        )

    second = tmp_path / "receipts" / "second.json"
    create_stage_a_launch_receipt(
        release.repo,
        second,
        authorization_commit=release.authorization,
        manifest_path=MANIFEST_PATH,
        created_at=T0,
    )
    with pytest.raises(ReleaseVerificationError, match="exactly one"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt, second],
            now=T0,
        )


def test_absent_and_stale_receipts_are_rejected(tmp_path: Path) -> None:
    release = valid_release(tmp_path)
    with pytest.raises(ReleaseVerificationError, match="exactly one"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[],
            now=T0,
        )
    with pytest.raises(ReleaseVerificationError, match="stale"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt],
            now=T0 + timedelta(seconds=301),
        )


def test_manifest_parent_blob_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = deepcopy(manifest_for(repo, root))
    manifest["inventory"][0]["sha256"] = "0" * 64
    authorization = commit_authorization(repo, manifest)
    with pytest.raises(ReleaseVerificationError, match="inventory differs"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_receipt_manifest_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    release = valid_release(tmp_path)
    receipt = json.loads(release.receipt.read_bytes())
    receipt["manifest_sha256"] = "0" * 64
    receipt["payload_sha256"] = receipt_payload_sha256(receipt)
    release.receipt.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(ReleaseVerificationError, match="manifest_sha256 binding differs"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt],
            now=T0,
        )


def test_preregistration_must_be_only_the_exact_status_line_change(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    parent_bytes = (repo / PREREGISTRATION).read_bytes()
    authorization = commit_authorization(
        repo,
        manifest,
        preregistration_bytes=parent_bytes.replace(OLD_STATUS, NEW_STATUS, 1)
        + b"extra amendment\n",
    )
    with pytest.raises(ReleaseVerificationError, match="exact configured one-line"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_manifest_bytes_must_be_canonical(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)
    noncanonical = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    authorization = commit_authorization(repo, manifest, manifest_bytes=noncanonical)
    with pytest.raises(ReleaseVerificationError, match="not canonical"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_merge_authorization_commit_is_rejected(tmp_path: Path) -> None:
    repo, root = init_static_root(tmp_path)
    manifest = manifest_for(repo, root)

    git(repo, "checkout", "-b", "side", root)
    write(repo, "side.txt", b"side parent\n")
    git(repo, "add", "side.txt")
    git(repo, "commit", "-m", "side parent")
    side = git(repo, "rev-parse", "HEAD")

    git(repo, "checkout", "trunk")
    ordinary_authorization = commit_authorization(repo, manifest)
    authorization_tree = git(repo, "rev-parse", f"{ordinary_authorization}^{{tree}}")
    merge_authorization = git(
        repo,
        "commit-tree",
        authorization_tree,
        "-p",
        root,
        "-p",
        side,
        input_text="synthetic two-parent authorization\n",
    )

    with pytest.raises(ReleaseVerificationError, match="exactly one parent"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=merge_authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_exact_head_mismatch_is_rejected_before_launch(tmp_path: Path) -> None:
    release = valid_release(tmp_path)
    git(release.repo, "checkout", "--detach", release.root)
    with pytest.raises(ReleaseVerificationError, match="differs from authorization commit"):
        verify_stage_a_checkout(
            release.repo,
            authorization_commit=release.authorization,
            manifest_path=MANIFEST_PATH,
            receipt_paths=[release.receipt],
            now=T0,
        )
