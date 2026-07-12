from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from powered_v13_release import (
    STAGE_T,
    STAGE_T_INVENTORY_CONTRACT_PATH,
    STAGE_T_MANIFEST_SCHEMA,
    STAGE_T_NEW_STATUS_LINE,
    STAGE_T_OLD_STATUS_LINE,
    STAGE_T_PREREGISTRATION_PATH,
    STAGE_T_RECEIPT_BASENAME,
    ReleaseVerificationError,
    build_stage_t_manifest,
    canonical_json_bytes,
    create_stage_t_launch_receipt,
    encode_stage_t_inventory_contract,
    encode_stage_t_manifest,
    receipt_payload_sha256,
    sha256_bytes,
    verify_stage_a_authorization_commit,
    verify_stage_t_authorization_commit,
    verify_stage_t_checkout,
    verify_stage_t_manifest_document,
)


MANIFEST_PATH = "release/powered-v13-technical-canary-manifest.json"
T0 = datetime(2026, 7, 12, 18, 0, 0, tzinfo=timezone.utc)
INVENTORY_PATHS = tuple(sorted((
    STAGE_T_PREREGISTRATION_PATH,
    STAGE_T_INVENTORY_CONTRACT_PATH,
    "locks/uv.lock",
    "src/subject_loader.py",
    "src/technical_runner.py",
)))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr}"
        )
    return result.stdout.strip()


def _write(repo: Path, relative: str, payload: bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _root(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "trunk")
    _git(repo, "config", "user.email", "technical-release@example.com")
    _git(repo, "config", "user.name", "Technical Release Test")
    _write(
        repo,
        STAGE_T_PREREGISTRATION_PATH,
        b"# Synthetic v13\n\n" + STAGE_T_OLD_STATUS_LINE + b"\n",
    )
    _write(repo, "src/subject_loader.py", b"MODEL = 'pinned'\n")
    _write(repo, "src/technical_runner.py", b"SEMANTIC_N = 0\n")
    _write(repo, "locks/uv.lock", b"exact-lock\n")
    _write(
        repo,
        STAGE_T_INVENTORY_CONTRACT_PATH,
        encode_stage_t_inventory_contract(INVENTORY_PATHS),
    )
    _git(repo, "add", *INVENTORY_PATHS)
    _git(repo, "commit", "-m", "technical root")
    return repo, _git(repo, "rev-parse", "HEAD")


def _manifest(repo: Path, root: str) -> dict:
    return build_stage_t_manifest(
        repo, static_root_commit=root, manifest_path=MANIFEST_PATH
    )


def _authorization(
    repo: Path,
    manifest: dict,
    *,
    preregistration: bytes | None = None,
    manifest_bytes: bytes | None = None,
    extras: dict[str, bytes] | None = None,
) -> str:
    parent_prereg = (repo / STAGE_T_PREREGISTRATION_PATH).read_bytes()
    _write(
        repo,
        STAGE_T_PREREGISTRATION_PATH,
        preregistration if preregistration is not None else
        parent_prereg.replace(
            STAGE_T_OLD_STATUS_LINE, STAGE_T_NEW_STATUS_LINE, 1
        ),
    )
    _write(
        repo,
        MANIFEST_PATH,
        manifest_bytes if manifest_bytes is not None else
        encode_stage_t_manifest(manifest),
    )
    changed = [STAGE_T_PREREGISTRATION_PATH, MANIFEST_PATH]
    for path, payload in (extras or {}).items():
        _write(repo, path, payload)
        changed.append(path)
    _git(repo, "add", *changed)
    _git(repo, "commit", "-m", "technical authorization")
    return _git(repo, "rev-parse", "HEAD")


def _valid_release(tmp_path: Path, *, detach: bool = True):
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    authorization = _authorization(repo, manifest)
    receipts = tmp_path / "receipts"
    create_stage_t_launch_receipt(
        repo,
        receipts,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        created_at=T0,
    )
    if detach:
        _git(repo, "checkout", "--detach", authorization)
    return repo, root, authorization, receipts, manifest


def test_stage_t_constants_and_parent_inventory_are_exact(tmp_path: Path) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    evidence = verify_stage_t_manifest_document(
        repo, manifest, expected_manifest_path=MANIFEST_PATH
    )
    assert STAGE_T == "TECHNICAL_CANARY"
    assert STAGE_T_OLD_STATUS_LINE == (
        b"**Status:** **DRAFT \xe2\x80\x94 NO PAID WORK OR PRIMARY TREATMENT AUTHORIZED**\n"
    )
    assert STAGE_T_NEW_STATUS_LINE == (
        b"**Status:** **TECHNICAL_CANARY_AUTHORIZED \xe2\x80\x94 "
        b"E01/LONG ONLY; SEMANTIC N=0**\n"
    )
    assert manifest["schema"] == STAGE_T_MANIFEST_SCHEMA
    assert [row["path"] for row in manifest["inventory"]] == \
        list(INVENTORY_PATHS)
    assert evidence["static_root_commit"] == root
    assert encode_stage_t_manifest(manifest) == \
        canonical_json_bytes(manifest) + b"\n"


def test_exact_detached_clean_stage_t_checkout_and_receipt_pass(
    tmp_path: Path,
) -> None:
    repo, root, authorization, receipts, _manifest_doc = _valid_release(tmp_path)
    evidence = verify_stage_t_checkout(
        repo,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        receipt_directory=receipts,
        now=T0 + timedelta(seconds=9),
    )
    assert evidence["status"] == "PASS"
    assert evidence["stage"] == STAGE_T
    assert evidence["authorization"]["static_root_commit"] == root
    assert evidence["receipt"]["age_seconds"] == 9


def test_stage_a_wrapper_cannot_accept_stage_t_release(tmp_path: Path) -> None:
    repo, _root, authorization, _receipts, _manifest_doc = _valid_release(
        tmp_path, detach=False
    )
    with pytest.raises(ReleaseVerificationError, match="Stage-A manifest identity"):
        verify_stage_a_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )


def test_wrong_stage_t_status_and_inventory_are_rejected(tmp_path: Path) -> None:
    repo, root = _root(tmp_path)
    wrong_status = deepcopy(_manifest(repo, root))
    wrong_status["status_transition"]["new_bytes_hex"] = (
        b"**Status:** **STATIC_FROZEN_PHASE_A_AUTHORIZED**\n".hex()
    )
    with pytest.raises(ReleaseVerificationError, match="fixed v13 transition"):
        verify_stage_t_manifest_document(repo, wrong_status)

    omitted = deepcopy(_manifest(repo, root))
    omitted["inventory"] = omitted["inventory"][:-1]
    omitted["inventory_count"] = len(omitted["inventory"])
    omitted["inventory_sha256"] = sha256_bytes(
        canonical_json_bytes(omitted["inventory"])
    )
    with pytest.raises(ReleaseVerificationError, match="fixed parent contract"):
        verify_stage_t_manifest_document(repo, omitted)


def test_wrong_stage_t_parent_hash_and_extra_diff_are_rejected(
    tmp_path: Path,
) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    _write(repo, "intervening.txt", b"intervening\n")
    _git(repo, "add", "intervening.txt")
    _git(repo, "commit", "-m", "intervening")
    authorization = _authorization(repo, manifest)
    with pytest.raises(ReleaseVerificationError, match="parent differs"):
        verify_stage_t_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )

    other = tmp_path / "other"
    other.mkdir()
    repo2, root2 = _root(other)
    corrupt = deepcopy(_manifest(repo2, root2))
    corrupt["inventory"][0]["sha256"] = "0" * 64
    corrupt_auth = _authorization(repo2, corrupt)
    with pytest.raises(ReleaseVerificationError, match="inventory differs"):
        verify_stage_t_authorization_commit(
            repo2,
            authorization_commit=corrupt_auth,
            manifest_path=MANIFEST_PATH,
        )

    third = tmp_path / "third"
    third.mkdir()
    repo3, root3 = _root(third)
    extra_auth = _authorization(
        repo3, _manifest(repo3, root3), extras={"extra.py": b"paid = True\n"}
    )
    with pytest.raises(ReleaseVerificationError, match="diff paths/statuses differ"):
        verify_stage_t_authorization_commit(
            repo3,
            authorization_commit=extra_auth,
            manifest_path=MANIFEST_PATH,
        )


@pytest.mark.parametrize("mode", ["branch", "dirty", "wrong-head"])
def test_stage_t_checkout_state_fails_closed(tmp_path: Path, mode: str) -> None:
    repo, root, authorization, receipts, _manifest_doc = _valid_release(
        tmp_path, detach=(mode != "branch")
    )
    if mode == "dirty":
        _write(repo, "untracked.txt", b"dirty\n")
    elif mode == "wrong-head":
        _git(repo, "checkout", "--detach", root)
    expected = {
        "branch": "on branch",
        "dirty": "checkout is dirty",
        "wrong-head": "differs from authorization commit",
    }[mode]
    with pytest.raises(ReleaseVerificationError, match=expected):
        verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0,
        )


def test_stage_t_receipt_mutation_duplicate_and_staleness_fail(
    tmp_path: Path,
) -> None:
    repo, _root, authorization, receipts, _manifest_doc = _valid_release(tmp_path)
    receipt_path = receipts / STAGE_T_RECEIPT_BASENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["manifest_sha256"] = "0" * 64
    receipt["payload_sha256"] = receipt_payload_sha256(receipt)
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(ReleaseVerificationError, match="manifest_sha256 binding differs"):
        verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0,
        )

    receipt_path.unlink()
    create_stage_t_launch_receipt(
        repo,
        receipts,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        created_at=T0,
    )
    _write(receipts, "extra.json", b"{}\n")
    with pytest.raises(ReleaseVerificationError, match="observed 2"):
        verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0,
        )
    (receipts / "extra.json").unlink()
    with pytest.raises(ReleaseVerificationError, match="stale"):
        verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0 + timedelta(seconds=301),
        )


def test_stage_t_preregistration_change_is_exactly_one_line(tmp_path: Path) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    parent = (repo / STAGE_T_PREREGISTRATION_PATH).read_bytes()
    authorization = _authorization(
        repo,
        manifest,
        preregistration=parent.replace(
            STAGE_T_OLD_STATUS_LINE, STAGE_T_NEW_STATUS_LINE, 1
        ) + b"extra authorization text\n",
    )
    with pytest.raises(ReleaseVerificationError, match="exact configured one-line"):
        verify_stage_t_authorization_commit(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
        )
