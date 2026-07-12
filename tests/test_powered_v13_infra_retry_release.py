from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.insert(0, str(SRC))

import powered_v13_infra_retry_entrypoint as entry  # noqa: E402
import powered_v13_infra_retry_release as release  # noqa: E402


MANIFEST_PATH = (
    "results/coherent_state_powered_v13/releases/"
    "powered-v13-stage-t-infra-retry-1-manifest_20260712T2310Z.json"
)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=True,
    ).stdout.strip()


def static_root() -> str:
    # This test is added in the static-root commit. It remains a stable locator
    # when the suite later runs from the two-path authorization child.
    return git("log", "-1", "--format=%H", "--", "tests/test_powered_v13_infra_retry_release.py")


def test_real_static_root_manifest_binds_retry_and_evidence() -> None:
    document = release.build_infra_retry_manifest(
        ROOT, static_root_commit=static_root(), manifest_path=MANIFEST_PATH
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


def test_manifest_tampering_fails_closed() -> None:
    document = release.build_infra_retry_manifest(
        ROOT, static_root_commit=static_root(), manifest_path=MANIFEST_PATH
    )
    tampered = deepcopy(document)
    tampered["retry_constraints"]["max_provider_allocations"] = 2
    with pytest.raises(release.InfraRetryReleaseError, match="constraints differ"):
        release.verify_infra_retry_manifest(ROOT, tampered)


def _args(tmp_path: Path) -> argparse.Namespace:
    inner_receipts = tmp_path / "inner-receipts"
    inner_receipts.mkdir()
    (inner_receipts / "powered-v13-technical-canary-launch-receipt.json").write_bytes(
        b"inner\n"
    )
    return argparse.Namespace(
        outer_authorization_commit="a" * 40,
        outer_manifest=MANIFEST_PATH,
        outer_receipt_directory=tmp_path / "outer-receipts",
        inner_repo=tmp_path / "inner-repo",
        inner_receipt_directory=inner_receipts,
        import_report=Path("data/coherent_state_powered_v13/technical-canary-import-audit-v1.json"),
        import_report_sha256="b" * 64,
        session_root=tmp_path / "session",
        ssh_key=tmp_path / "ssh-key",
        hf_token=tmp_path / "hf-token",
        primary_batch_id="stage-t-infra-retry-1-test",
    )


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


def test_delegate_command_has_unoverrideable_carry_ins_and_one_attempt(
    tmp_path: Path, monkeypatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(entry, "verify_infra_retry_checkout", lambda *_a, **_k: {})
    monkeypatch.setattr(entry, "verify_stage_t_checkout", lambda *_a, **_k: {})
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
