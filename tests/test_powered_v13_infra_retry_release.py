from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

import powered_v13_infra_retry_entrypoint as entry  # noqa: E402
import powered_v13_infra_retry_release as release  # noqa: E402
from powered_v13_release import create_stage_t_launch_receipt  # noqa: E402


MANIFEST_PATH = (
    "results/coherent_state_powered_v13/releases/"
    "powered-v13-stage-t-infra-retry-1-manifest_20260712T2310Z.json"
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def repo_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def make_static_repo(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(ROOT), str(repo)],
        check=True,
    )
    repo_git(repo, "checkout", "-q", "--detach", release.ONE_ATTEMPT_CONTROLLER_COMMIT)
    repo_git(repo, "config", "user.email", "test@example.invalid")
    repo_git(repo, "config", "user.name", "Release Test")
    release_paths = (
        release.AMENDMENT_PATH,
        release.CONTRACT_PATH,
        "scripts/run_powered_v13_stage_t_infra_retry_1.py",
        "src/powered_v13_infra_retry_entrypoint.py",
        "src/powered_v13_infra_retry_release.py",
    )
    for path in release_paths:
        destination = repo / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = (ROOT / path).read_bytes()
        if path == release.AMENDMENT_PATH and release.NEW_STATUS_LINE in raw:
            raw = raw.replace(release.NEW_STATUS_LINE, release.OLD_STATUS_LINE, 1)
        destination.write_bytes(raw)
    repo_git(repo, "add", *release_paths)
    repo_git(repo, "commit", "-q", "-m", "synthetic retry static root")
    return repo, repo_git(repo, "rev-parse", "HEAD")


def make_authorization_repo(
    tmp_path: Path,
    *,
    extra_path: bool = False,
    wrong_status: bool = False,
    wrong_manifest: bool = False,
) -> tuple[Path, str]:
    repo, static_root_commit = make_static_repo(tmp_path)
    amendment = repo / release.AMENDMENT_PATH
    raw = amendment.read_bytes()
    replacement = release.NEW_STATUS_LINE
    if wrong_status:
        replacement = b"**Status:** **INFRA_RETRY_AUTHORIZED WITH EXTRA CHANGE**\n"
    amendment.write_bytes(raw.replace(release.OLD_STATUS_LINE, replacement, 1))
    manifest = release.build_infra_retry_manifest(
        repo, static_root_commit=static_root_commit, manifest_path=MANIFEST_PATH
    )
    if wrong_manifest:
        manifest["retry_constraints"]["max_provider_allocations"] = 2
    manifest_raw = release.encode_infra_retry_manifest(manifest)
    manifest_path = repo / MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_raw)
    paths = [release.AMENDMENT_PATH, MANIFEST_PATH]
    if extra_path:
        extra = repo / "unexpected-provider-authority.txt"
        extra.write_text("unexpected\n")
        paths.append("unexpected-provider-authority.txt")
    repo_git(repo, "add", *paths)
    repo_git(repo, "commit", "-q", "-m", "synthetic retry authorization")
    return repo, repo_git(repo, "rev-parse", "HEAD")


def test_real_static_root_manifest_binds_retry_and_evidence(tmp_path: Path) -> None:
    repo, static_root_commit = make_static_repo(tmp_path)
    document = release.build_infra_retry_manifest(
        repo, static_root_commit=static_root_commit, manifest_path=MANIFEST_PATH
    )
    assert document["inventory_count"] == len(release.FIXED_INVENTORY_PATHS)
    assert document["original_scientific_payload"] == {
        "authorization_commit": release.ORIGINAL_AUTHORIZATION_COMMIT,
        "manifest_path": release.ORIGINAL_MANIFEST_PATH,
        "manifest_sha256": release.ORIGINAL_MANIFEST_SHA256,
        "semantic_n": 0,
        "static_root_commit": "f3d4ed0a62f93aa5c5a601ef95a4eaa550777a85",
    }
    assert document["provider_adapter"] == {
        "controller_paths": list(release.CONTROLLER_PATHS),
        "max_allocation_attempts": 1,
        "one_attempt_controller_commit": release.ONE_ATTEMPT_CONTROLLER_COMMIT,
        "patch_commit": release.PROVIDER_PATCH_COMMIT,
        "repair_scope": "runpod-real-allocation-response-schema-only",
    }
    constraints = document["retry_constraints"]
    assert constraints["max_provider_allocations"] == 1
    assert constraints["no_fallback"] is True
    assert constraints["outer_authorization_is_sole_provider_authority"] is True
    assert constraints["inner_stage_t_receipt_provider_authority"] is False
    assert constraints["carry_in_provider_seconds"] == 58
    assert constraints["carry_in_conservative_spend_usd"] == (
        "0.02239444444444444444444444444"
    )
    assert release.encode_infra_retry_manifest(document) == (
        release.canonical_json_bytes(document) + b"\n"
    )


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    repo, static_root_commit = make_static_repo(tmp_path)
    document = release.build_infra_retry_manifest(
        repo, static_root_commit=static_root_commit, manifest_path=MANIFEST_PATH
    )
    tampered = deepcopy(document)
    tampered["retry_constraints"]["max_provider_allocations"] = 2
    with pytest.raises(release.InfraRetryReleaseError, match="constraints differ"):
        release.verify_infra_retry_manifest(repo, tampered)


def test_real_two_path_authorization_and_fresh_receipt_pass(tmp_path: Path) -> None:
    repo, authorization = make_authorization_repo(tmp_path)
    evidence = release.verify_infra_retry_authorization(
        repo, authorization_commit=authorization, manifest_path=MANIFEST_PATH
    )
    assert evidence["static_root_commit"] == repo_git(
        repo, "rev-parse", f"{authorization}^"
    )
    assert evidence["changed_paths"] == sorted((release.AMENDMENT_PATH, MANIFEST_PATH))
    receipts = tmp_path / "receipts"
    t0 = datetime(2026, 7, 12, 23, 10, tzinfo=timezone.utc)
    release.create_infra_retry_receipt(
        repo, receipts, authorization_commit=authorization,
        manifest_path=MANIFEST_PATH, created_at=t0,
    )
    verified = release.verify_infra_retry_checkout(
        repo, authorization_commit=authorization, manifest_path=MANIFEST_PATH,
        receipt_directory=receipts, now=t0 + timedelta(seconds=2),
    )
    assert verified["status"] == "PASS"
    assert verified["receipt"]["age_seconds"] == 2


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"extra_path": True}, "two-path"),
        ({"wrong_status": True}, "status-only"),
        ({"wrong_manifest": True}, "constraints differ"),
    ],
)
def test_authorization_extra_path_status_or_manifest_fails_closed(
    tmp_path: Path, mutation: dict[str, bool], match: str,
) -> None:
    repo, authorization = make_authorization_repo(tmp_path, **mutation)
    with pytest.raises(release.InfraRetryReleaseError, match=match):
        release.verify_infra_retry_authorization(
            repo, authorization_commit=authorization, manifest_path=MANIFEST_PATH
        )


def test_stale_attached_dirty_and_extra_receipt_checkout_fail_closed(
    tmp_path: Path,
) -> None:
    repo, authorization = make_authorization_repo(tmp_path)
    receipts = tmp_path / "receipts"
    t0 = datetime(2026, 7, 12, 23, 10, tzinfo=timezone.utc)
    release.create_infra_retry_receipt(
        repo, receipts, authorization_commit=authorization,
        manifest_path=MANIFEST_PATH, created_at=t0,
    )
    kwargs = {
        "authorization_commit": authorization,
        "manifest_path": MANIFEST_PATH,
        "receipt_directory": receipts,
        "now": t0 + timedelta(seconds=1),
    }
    with pytest.raises(release.InfraRetryReleaseError, match="stale"):
        release.verify_infra_retry_checkout(
            repo, **{**kwargs, "now": t0 + timedelta(seconds=301)}
        )
    repo_git(repo, "checkout", "-q", "-b", "attached-test")
    with pytest.raises(release.InfraRetryReleaseError, match="on a branch"):
        release.verify_infra_retry_checkout(repo, **kwargs)
    repo_git(repo, "checkout", "-q", "--detach", authorization)
    (repo / release.AMENDMENT_PATH).write_bytes(
        (repo / release.AMENDMENT_PATH).read_bytes() + b"dirty\n"
    )
    with pytest.raises(release.InfraRetryReleaseError, match="dirty"):
        release.verify_infra_retry_checkout(repo, **kwargs)
    repo_git(repo, "restore", release.AMENDMENT_PATH)
    (receipts / "extra.json").write_text("{}\n")
    with pytest.raises(release.InfraRetryReleaseError, match="observed 2"):
        release.verify_infra_retry_checkout(repo, **kwargs)


def test_noncanonical_and_tampered_receipts_fail_closed(tmp_path: Path) -> None:
    repo, authorization = make_authorization_repo(tmp_path)
    receipts = tmp_path / "receipts"
    t0 = datetime(2026, 7, 12, 23, 10, tzinfo=timezone.utc)
    receipt_path = release.create_infra_retry_receipt(
        repo, receipts, authorization_commit=authorization,
        manifest_path=MANIFEST_PATH, created_at=t0,
    )
    receipt = json.loads(receipt_path.read_bytes())
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with pytest.raises(release.InfraRetryReleaseError, match="not canonical"):
        release.verify_infra_retry_checkout(
            repo, authorization_commit=authorization, manifest_path=MANIFEST_PATH,
            receipt_directory=receipts, now=t0,
        )
    receipt["manifest_sha256"] = "0" * 64
    receipt["payload_sha256"] = release._receipt_payload_sha256(receipt)
    receipt_path.write_bytes(release.canonical_json_bytes(receipt) + b"\n")
    with pytest.raises(release.InfraRetryReleaseError, match="manifest binding differs"):
        release.verify_infra_retry_checkout(
            repo, authorization_commit=authorization, manifest_path=MANIFEST_PATH,
            receipt_directory=receipts, now=t0,
        )


def _args(tmp_path: Path) -> argparse.Namespace:
    inner_repo = tmp_path / "inner-repo"
    inner_repo.mkdir()
    inner_receipts = tmp_path / "inner-receipts"
    inner_receipts.mkdir()
    (inner_receipts / "powered-v13-technical-canary-launch-receipt.json").write_bytes(
        b"inner\n"
    )
    outer_receipts = tmp_path / "outer-receipts"
    outer_receipts.mkdir()
    (outer_receipts / release.RECEIPT_BASENAME).write_bytes(b"outer\n")
    import_report = inner_repo / (
        "data/coherent_state_powered_v13/technical-canary-import-audit-v1.json"
    )
    import_report.parent.mkdir(parents=True)
    import_report.write_bytes(b"report\n")
    batch_id = f"stage-t-infra-retry-1-{'a' * 12}-20260712T231000Z"
    return argparse.Namespace(
        outer_authorization_commit="a" * 40,
        outer_manifest=MANIFEST_PATH,
        outer_receipt_directory=outer_receipts,
        inner_repo=inner_repo,
        inner_receipt_directory=inner_receipts,
        import_report=Path("data/coherent_state_powered_v13/technical-canary-import-audit-v1.json"),
        import_report_sha256="b" * 64,
        session_root=tmp_path / batch_id,
        ssh_key=tmp_path / "ssh-key",
        hf_token=tmp_path / "hf-token",
        primary_batch_id=batch_id,
    )


def _mock_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(entry, "CONSUMPTION_ROOT", tmp_path / "fixed-consumption-root")
    monkeypatch.setattr(entry, "verify_infra_retry_checkout", lambda *_a, **_k: {
        "authorization": {
            "manifest_sha256": "c" * 64,
            "static_root_commit": "d" * 40,
        },
        "receipt": {"authorization_commit": "a" * 40},
    })
    monkeypatch.setattr(entry, "verify_stage_t_checkout", lambda *_a, **_k: {
        "authorization": {"manifest_sha256": "e" * 64},
        "receipt": {
            "receipt_sha256": release.sha256_bytes(b"inner\n"),
        },
    })
    monkeypatch.setattr(entry, "verify_release_binding", lambda **_k: object())


def test_outer_mismatch_never_reaches_delegate(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    called = False

    def fail_outer(*_args, **_kwargs):
        raise release.InfraRetryReleaseError("outer mismatch")

    def delegate(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("delegate reached")

    monkeypatch.setattr(entry, "verify_infra_retry_checkout", fail_outer)
    with pytest.raises(release.InfraRetryReleaseError, match="outer mismatch"):
        entry.run(args, delegate=delegate)
    assert called is False


def test_inner_mismatch_never_reaches_delegate(tmp_path: Path, monkeypatch) -> None:
    args = _args(tmp_path)
    called = False
    monkeypatch.setattr(entry, "verify_infra_retry_checkout", lambda *_a, **_k: {})

    def fail_inner(*_args, **_kwargs):
        raise RuntimeError("inner mismatch")

    def delegate(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("delegate reached")

    monkeypatch.setattr(entry, "verify_stage_t_checkout", fail_inner)
    with pytest.raises(RuntimeError, match="inner mismatch"):
        entry.run(args, delegate=delegate)
    assert called is False


@pytest.mark.parametrize(
    "violation", ["batch", "basename", "outer_session", "inner_session"]
)
def test_batch_or_session_boundary_violation_never_reaches_delegate(
    tmp_path: Path, monkeypatch, violation: str,
) -> None:
    args = _args(tmp_path)
    if violation == "batch":
        args.primary_batch_id = "stage-t-infra-retry-1-bbbbbbbbbbbb-20260712T231000Z"
    elif violation == "basename":
        args.session_root = tmp_path / "wrong-basename"
    elif violation == "outer_session":
        args.session_root = entry.EXECUTING_ROOT / "forbidden-session"
    else:
        args.session_root = args.inner_repo / "forbidden-session"
    called = False

    def delegate(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("delegate reached")

    monkeypatch.setattr(entry, "verify_infra_retry_checkout", lambda *_a, **_k: {})
    monkeypatch.setattr(entry, "verify_stage_t_checkout", lambda *_a, **_k: {})
    with pytest.raises(entry.InfraRetryEntrypointError):
        entry.run(args, delegate=delegate)
    assert called is False


def test_delegate_command_has_unoverrideable_carry_ins_and_one_attempt(
    tmp_path: Path, monkeypatch,
) -> None:
    args = _args(tmp_path)
    _mock_success(monkeypatch, tmp_path)
    observed = []

    def delegate(command, *, check):
        observed.append((command, check))
        return subprocess.CompletedProcess(command, 0)

    assert entry.run(args, delegate=delegate) == 0
    assert len(observed) == 1 and observed[0][1] is False
    command = observed[0][0]
    assert command[command.index("--prior-stage-t-spend-usd") + 1] == (
        release.CARRY_IN_CONSERVATIVE_SPEND_USD
    )
    assert command[command.index("--prior-stage-t-provider-seconds") + 1] == "58"
    assert command[command.index("--max-allocation-attempts") + 1] == "1"
    assert command[command.index("--authorization-commit") + 1] == (
        release.ORIGINAL_AUTHORIZATION_COMMIT
    )
    assert command[command.index("--repository-url") + 1] == entry.REPOSITORY_URL
    marker = entry.CONSUMPTION_ROOT / f"{args.outer_authorization_commit}.json"
    record = json.loads(marker.read_bytes())
    assert record["outer_authorization_commit"] == args.outer_authorization_commit
    assert record["inner_authorization_commit"] == release.ORIGINAL_AUTHORIZATION_COMMIT
    assert record["batch_id"] == args.primary_batch_id
    assert record["session_root"] == str(args.session_root.resolve())
    assert record["max_allocation_attempts"] == 1


def test_authorization_consumption_blocks_second_delegate_forever(
    tmp_path: Path, monkeypatch,
) -> None:
    outer_repo, outer_authorization = make_authorization_repo(tmp_path / "outer")
    outer_receipts = tmp_path / "outer-receipts"
    release.create_infra_retry_receipt(
        outer_repo, outer_receipts, authorization_commit=outer_authorization,
        manifest_path=MANIFEST_PATH,
    )
    inner_repo = tmp_path / "inner-repo"
    subprocess.run(
        ["git", "clone", "-q", "--no-hardlinks", str(ROOT), str(inner_repo)],
        check=True,
    )
    repo_git(inner_repo, "checkout", "-q", "--detach", release.ORIGINAL_AUTHORIZATION_COMMIT)
    inner_receipts = tmp_path / "inner-receipts"
    create_stage_t_launch_receipt(
        inner_repo, inner_receipts,
        authorization_commit=release.ORIGINAL_AUTHORIZATION_COMMIT,
        manifest_path=release.ORIGINAL_MANIFEST_PATH,
    )
    report = inner_repo / "data/coherent_state_powered_v13/technical-canary-import-audit-v1.json"
    batch_id = (
        f"stage-t-infra-retry-1-{outer_authorization[:12]}-20260712T231000Z"
    )
    args = argparse.Namespace(
        outer_authorization_commit=outer_authorization,
        outer_manifest=MANIFEST_PATH,
        outer_receipt_directory=outer_receipts,
        inner_repo=inner_repo,
        inner_receipt_directory=inner_receipts,
        import_report=report,
        import_report_sha256=release.sha256_bytes(report.read_bytes()),
        session_root=tmp_path / batch_id,
        ssh_key=tmp_path / "ssh-key",
        hf_token=tmp_path / "hf-token",
        primary_batch_id=batch_id,
    )
    monkeypatch.setattr(entry, "EXECUTING_ROOT", outer_repo)
    monkeypatch.setattr(entry, "CONSUMPTION_ROOT", tmp_path / "fixed-consumption-root")
    calls = []

    def delegate(command, *, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 9)

    assert entry.run(args, delegate=delegate) == 9
    with pytest.raises(entry.InfraRetryEntrypointError, match="already consumed"):
        entry.run(args, delegate=delegate)
    assert len(calls) == 1
    marker = entry.CONSUMPTION_ROOT / f"{outer_authorization}.json"
    assert marker.is_file()
    record = json.loads(marker.read_bytes())
    assert marker.read_bytes() == release.canonical_json_bytes(record) + b"\n"


def test_parser_exposes_no_outer_root_or_budget_override(tmp_path: Path) -> None:
    base = [
        "--outer-authorization-commit", "a" * 40,
        "--outer-manifest", MANIFEST_PATH,
        "--outer-receipt-directory", str(tmp_path / "outer"),
        "--inner-repo", str(tmp_path / "inner"),
        "--inner-receipt-directory", str(tmp_path / "inner-receipt"),
        "--import-report", "report.json", "--import-report-sha256", "b" * 64,
        "--session-root", str(tmp_path / "session"), "--ssh-key", "ssh",
        "--hf-token", "hf", "--primary-batch-id", "batch",
    ]
    entry.parser().parse_args(base)
    for forbidden in (
        ["--outer-repo", str(tmp_path / "decoy")],
        ["--prior-stage-t-provider-seconds", "0"],
        ["--prior-stage-t-spend-usd", "0"],
        ["--max-allocation-attempts", "2"],
        ["--repository-url", "https://example.invalid/decoy.git"],
    ):
        with pytest.raises(SystemExit) as rejected:
            entry.parser().parse_args([*base, *forbidden])
        assert rejected.value.code == 2
