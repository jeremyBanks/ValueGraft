from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import inspect
import json
from pathlib import Path
import subprocess

import pytest

import powered_v13_release as release
from powered_v13_release import (
    STAGE_T,
    STAGE_T_INVENTORY_CONTRACT_PATH,
    STAGE_T_MANIFEST_SCHEMA,
    STAGE_T_NEW_STATUS_LINE,
    STAGE_T_OLD_STATUS_LINE,
    STAGE_T_PREREGISTRATION_PATH,
    STAGE_T_RECEIPT_BASENAME,
    ReleaseVerificationError,
    _create_stage_t_launch_receipt_for_test as create_stage_t_launch_receipt,
    _verify_stage_t_checkout_for_test as verify_stage_t_checkout,
    build_stage_t_manifest,
    canonical_json_bytes,
    encode_stage_t_inventory_contract,
    encode_stage_t_manifest,
    receipt_payload_sha256,
    sha256_bytes,
    verify_stage_a_authorization_commit,
    verify_stage_t_authorization_commit,
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


def _root(tmp_path: Path, *, object_format: str = "sha1") -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    init_args = ["init", "-b", "trunk"]
    if object_format == "sha256":
        init_args = ["init", "--object-format=sha256", "-b", "trunk"]
    _git(repo, *init_args)
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


def _valid_release(
    tmp_path: Path, *, detach: bool = True, object_format: str = "sha1"
):
    repo, root = _root(tmp_path, object_format=object_format)
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
    with pytest.raises(
        ReleaseVerificationError, match="Stage-A manifest (?:fields|identity)"
    ):
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


def test_remote_setup_accepts_stale_exact_receipt_only_and_rejects_future(
    tmp_path: Path,
) -> None:
    repo, _root, authorization, receipts, _manifest_doc = _valid_release(
        tmp_path)
    stale_at = T0 + timedelta(days=1)
    setup = release._verify_stage_t_remote_setup_checkout_for_test(
        repo,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        receipt_directory=receipts,
        now=stale_at,
    )
    assert setup["status"] == "PASS"
    assert setup["receipt"]["age_seconds"] == 86400
    assert setup["receipt"]["receipt_role"] == "REMOTE_SETUP_ONLY"
    assert setup["receipt"]["age_gate"] == \
        "LOCAL_PREALLOCATION_AND_FRESH_SUBJECT_RECEIPT"

    with pytest.raises(ReleaseVerificationError, match="stale"):
        verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=stale_at,
        )
    with pytest.raises(ReleaseVerificationError, match="too far in the future"):
        release._verify_stage_t_remote_setup_checkout_for_test(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0 - timedelta(seconds=6),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "forged-stage-t-receipt-v1", "identity differs"),
        ("authorization_commit", "0" * 40, "authorization_commit binding differs"),
        ("static_root_commit", "0" * 40, "static_root_commit binding differs"),
        ("manifest_path", "release/other.json", "manifest_path binding differs"),
        ("manifest_sha256", "0" * 64, "manifest_sha256 binding differs"),
    ],
)
def test_remote_setup_rejects_wrong_receipt_schema_and_release_bindings(
    tmp_path: Path, field: str, value: str, message: str,
) -> None:
    repo, _root, authorization, receipts, _manifest_doc = _valid_release(
        tmp_path)
    receipt_path = receipts / STAGE_T_RECEIPT_BASENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt[field] = value
    receipt["payload_sha256"] = receipt_payload_sha256(receipt)
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(ReleaseVerificationError, match=message):
        release._verify_stage_t_remote_setup_checkout_for_test(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0 + timedelta(days=1),
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


def test_replacement_ref_attack_is_rejected_before_authorization_read(
    tmp_path: Path,
) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    valid_authorization = _authorization(repo, manifest)
    receipts = tmp_path / "receipts"
    create_stage_t_launch_receipt(
        repo,
        receipts,
        authorization_commit=valid_authorization,
        manifest_path=MANIFEST_PATH,
        created_at=T0,
    )
    root_tree = _git(repo, "rev-parse", f"{root}^{{tree}}")
    bad_authorization = _git(
        repo, "commit-tree", root_tree, "-p", root, "-m", "bad authorization"
    )
    _git(repo, "replace", bad_authorization, valid_authorization)

    receipt_path = receipts / STAGE_T_RECEIPT_BASENAME
    receipt = json.loads(receipt_path.read_bytes())
    receipt["authorization_commit"] = bad_authorization
    receipt["payload_sha256"] = receipt_payload_sha256(receipt)
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    _git(repo, "checkout", "--detach", bad_authorization)

    with pytest.raises(ReleaseVerificationError, match="replacement refs"):
        verify_stage_t_checkout(
            repo,
            authorization_commit=bad_authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
            now=T0,
        )


def test_legacy_graft_metadata_is_rejected(tmp_path: Path) -> None:
    repo, root = _root(tmp_path)
    git_dir = Path(_git(repo, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    graft = git_dir / "info" / "grafts"
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text(f"{root}\n", encoding="ascii")
    with pytest.raises(ReleaseVerificationError, match="graft metadata"):
        _manifest(repo, root)


def test_inherited_git_repository_selection_environment_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, root = _root(tmp_path)
    expected = _manifest(repo, root)
    poison = str(tmp_path / "ambient-git-override")
    for name in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_SHALLOW_FILE",
        "GIT_REPLACE_REF_BASE",
    ):
        monkeypatch.setenv(name, poison)
    monkeypatch.setenv("GIT_GLOB_PATHSPECS", "1")

    assert _manifest(repo, root) == expected


def test_generic_engine_rejects_forged_custom_spec_and_stage_key(
    tmp_path: Path,
) -> None:
    repo, root = _root(tmp_path)
    forged = release._ReleaseSpec(
        label="Weak",
        stage="WEAK",
        manifest_schema="weak-manifest-v1",
        inventory_contract_schema="weak-contract-v1",
        receipt_schema="weak-receipt-v1",
        preregistration_path="PROTOCOL.md",
        inventory_contract_path="data/weak.json",
        receipt_basename="weak.json",
        old_status_line=b"OLD\n",
        new_status_line=b"NEW\n",
    )
    with pytest.raises(ReleaseVerificationError, match="frozen built-in key"):
        release._build_manifest(
            repo,
            static_root_commit=root,
            manifest_path=MANIFEST_PATH,
            stage_key=forged,
        )
    with pytest.raises(ReleaseVerificationError, match="frozen built-in key"):
        release._spec_for("technical-canary")

    for function in (
        release.build_stage_t_manifest,
        release.verify_stage_t_authorization_commit,
        release.create_stage_t_launch_receipt,
        release.verify_stage_t_checkout,
    ):
        parameters = inspect.signature(function).parameters
        assert not ({"stage", "stage_key", "spec", "capability"} & set(parameters))


def test_production_receipt_apis_expose_no_clock_or_tolerance_knobs() -> None:
    forbidden = {
        "created_at", "now", "max_age_seconds", "max_receipt_age_seconds",
        "future_skew_seconds", "allow_stale", "skip_age", "enforce_age",
        "age_policy", "verify_checkout",
    }
    for function in (
        release.create_stage_t_launch_receipt,
        release.create_stage_a_launch_receipt,
        release.verify_stage_t_launch_receipt,
        release.verify_stage_a_launch_receipt,
        release.verify_stage_t_checkout,
        release.verify_stage_a_checkout,
    ):
        assert not (forbidden & set(inspect.signature(function).parameters))
    assert list(inspect.signature(
        release._verify_stage_t_remote_setup_checkout).parameters) == [
            "repo", "authorization_commit", "manifest_path",
            "receipt_directory",
        ]


def test_public_stage_t_receipt_and_checkout_use_real_clock_and_pass(
    tmp_path: Path,
) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    authorization = _authorization(repo, manifest)
    receipts = tmp_path / "receipts"
    release.create_stage_t_launch_receipt(
        repo,
        receipts,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
    )
    _git(repo, "checkout", "--detach", authorization)
    evidence = release.verify_stage_t_checkout(
        repo,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        receipt_directory=receipts,
    )
    assert evidence["status"] == "PASS"
    assert 0 <= evidence["receipt"]["age_seconds"] <= 5


@pytest.mark.parametrize(
    ("created_at", "message"),
    [
        (datetime(2000, 1, 1, tzinfo=timezone.utc), "stale"),
        (datetime(2100, 1, 1, tzinfo=timezone.utc), "too far in the future"),
    ],
)
def test_public_checkout_rejects_ancient_and_future_receipts_without_knobs(
    tmp_path: Path, created_at: datetime, message: str,
) -> None:
    repo, root = _root(tmp_path)
    manifest = _manifest(repo, root)
    authorization = _authorization(repo, manifest)
    receipts = tmp_path / "receipts"
    create_stage_t_launch_receipt(
        repo,
        receipts,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        created_at=created_at,
    )
    _git(repo, "checkout", "--detach", authorization)
    with pytest.raises(ReleaseVerificationError, match=message):
        release.verify_stage_t_checkout(
            repo,
            authorization_commit=authorization,
            manifest_path=MANIFEST_PATH,
            receipt_directory=receipts,
        )


@pytest.mark.parametrize(
    ("schema", "tag"),
    [
        (STAGE_T_MANIFEST_SCHEMA, "duplicate-stage-t"),
        (release.STAGE_A_MANIFEST_SCHEMA, "cross-stage-a"),
    ],
)
def test_stage_t_root_rejects_preexisting_duplicate_or_cross_stage_manifest(
    tmp_path: Path, schema: str, tag: str,
) -> None:
    repo, _root_commit = _root(tmp_path)
    extra = f"release/{tag}.json"
    _write(repo, extra, canonical_json_bytes({"schema": schema}) + b"\n")
    paths = tuple(sorted((*INVENTORY_PATHS, extra)))
    _write(
        repo,
        STAGE_T_INVENTORY_CONTRACT_PATH,
        encode_stage_t_inventory_contract(paths),
    )
    _git(repo, "add", extra, STAGE_T_INVENTORY_CONTRACT_PATH)
    _git(repo, "commit", "-m", "forbidden preexisting release artifact")
    contaminated_root = _git(repo, "rev-parse", "HEAD")
    with pytest.raises(
        ReleaseVerificationError, match="duplicate or cross-stage"
    ):
        _manifest(repo, contaminated_root)


def test_sha256_git_repository_still_passes_full_stage_t_gate(
    tmp_path: Path,
) -> None:
    repo, root, authorization, receipts, _manifest_doc = _valid_release(
        tmp_path, object_format="sha256"
    )
    evidence = verify_stage_t_checkout(
        repo,
        authorization_commit=authorization,
        manifest_path=MANIFEST_PATH,
        receipt_directory=receipts,
        now=T0,
    )
    assert evidence["status"] == "PASS"
    assert len(root) == len(authorization) == 64
    assert len(evidence["authorization"]["static_root_tree"]) == 64
