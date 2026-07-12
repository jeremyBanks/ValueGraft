from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest

import powered_v13_lifecycle as life


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_powered_v13_stage_t_lifecycle.py"


PRESERVED_CREATE_RESPONSE_SHAPE = {
    "id": "pod_stage_t_preserved",
    "costPerHr": "1.2",
    "gpuCount": 1,
    "machine": {
        "dataCenterId": "CA-MTL-1",
        "gpuTypeId": "NVIDIA A100 80GB PCIe",
        "secureCloud": True,
        "supportPublicIp": True,
    },
    "publicIp": "203.0.113.8",
}


def _provider_response(pod_id: str, **extra):
    response = deepcopy(PRESERVED_CREATE_RESPONSE_SHAPE)
    response["id"] = pod_id
    response.update(extra)
    return response


@pytest.fixture
def preserved_create_response():
    return deepcopy(PRESERVED_CREATE_RESPONSE_SHAPE)


class Clock:
    def __init__(self):
        self.value = 1_000_000

    def __call__(self):
        self.value += 1
        return self.value


class FakeProvider:
    def __init__(self, clock, actions=("success",), log=None,
                 response_mutation=None):
        self.clock = clock
        self.actions = list(actions)
        self.log = log if log is not None else []
        self.creates = 0
        self.active = []
        self.deleted = set()
        self.response_mutation = response_mutation

    def safe_snapshot(self):
        self.log.append("snapshot")
        return {"balance_usd": "57.12", "spend_limit_usd": "80",
                "active_pod_ids": sorted(self.active)}

    def create_secure_a100(self):
        self.creates += 1
        self.log.append(("create", self.creates, self.clock.value))
        action = self.actions.pop(0)
        if action == "no_capacity":
            raise life.NoCapacity("none")
        if action == "ambiguous":
            raise OSError("connection reset after POST")
        pod_id = f"pod_stage_t_{self.creates}"
        response = _provider_response(pod_id)
        if self.response_mutation is not None:
            self.response_mutation(response)
        self.active.append(pod_id)
        return life.Allocation(
            response=response,
            state_bytes=json.dumps({
                "id": pod_id, "costPerHr": "1.2",
            }, sort_keys=True).encode(),
        )

    def get_pod(self, pod_id):
        self.log.append(("get", pod_id))
        if pod_id in self.deleted:
            raise life.PodNotFound(pod_id)
        return {"id": pod_id}

    def active_pod_ids(self):
        self.log.append("inventory")
        return sorted(self.active)

    def delete_pod(self, pod_id):
        self.log.append(("delete", pod_id))
        self.deleted.add(pod_id)
        if pod_id in self.active:
            self.active.remove(pod_id)


class FakeTransport:
    def __init__(self, admissions=(True,), *, sync_error=False,
                 launch_error=False, log=None):
        self.admissions = list(admissions)
        self.sync_error = sync_error
        self.launch_error = launch_error
        self.log = log if log is not None else []
        self.synced = []

    def admit(self, allocation, *, timeout_seconds):
        self.log.append(("admit", allocation.pod_id, timeout_seconds))
        allowed = self.admissions.pop(0)
        if not allowed:
            raise life.AdmissionRejected("wrong host")
        return {
            "gpu_name": "NVIDIA A100 80GB PCIe", "memory_mib": 81920,
            "cuda_available": True, "gpu_count": 1,
            "torch_gpu_name": "NVIDIA A100 80GB PCIe",
            "gpu_uuid": "GPU-fixed", "driver_version": "580.65.06",
        }

    def sync(self, allocation, *, repo, relative_paths, external_files,
             timeout_seconds):
        self.log.append(("sync", allocation.pod_id))
        if self.sync_error:
            raise OSError("rsync failed")
        self.synced = list(relative_paths)
        assert all(Path(path).is_file() for path in external_files)
        return {"status": "PASS", "relative_paths": list(relative_paths),
                "external_file_count": len(external_files)}

    def launch_detached(self, allocation, *, job_path,
                        authorization_commit, timeout_seconds):
        self.log.append(("launch", allocation.pod_id, job_path))
        if self.launch_error:
            raise OSError("SSH pipe did not close")
        return {"status": "PASS", "pipe_eof": True, "child_alive": True,
                "pid": 4321, "job_path": job_path}


class FakeSupervisor:
    def __init__(self, provider, *, terminal_status="TERMINATED_HARVESTED",
                 reason="job_complete", dual=True, death=None,
                 start_errors=(), terminal_advance_seconds=0, log=None):
        self.provider = provider
        self.terminal_status = terminal_status
        self.reason = reason
        self.dual = dual
        self.death = death
        self.start_errors = list(start_errors)
        self.terminal_advance_seconds = terminal_advance_seconds
        self.log = log if log is not None else []
        self.harvests = 0
        self.cleanup_requests = 0

    def start(self, *, record_path, pod_state_path, lifecycle_state_path):
        assert record_path.is_file()
        assert pod_state_path.is_file()
        assert lifecycle_state_path.is_file()
        self.log.append(("watcher_start", record_path.name))
        if self.start_errors and self.start_errors.pop(0):
            raise OSError("watcher start failed")
        return {"record_path": record_path}

    def request_cleanup(self, handle, *, reason):
        self.cleanup_requests += 1
        self.log.append(("cleanup_request", reason))

    def wait_terminal(self, handle, *, timeout_seconds):
        self.log.append(("wait", timeout_seconds))
        self.provider.clock.value += self.terminal_advance_seconds
        if self.death:
            self.log.append(("emergency_cleanup", self.death))
        self.harvests += 1 if self.death != "after_harvest" else 0
        if self.death == "after_harvest" and self.harvests == 0:
            self.harvests = 1
        pod_id = next(iter(self.provider.active), None)
        if pod_id is not None:
            self.provider.delete_pod(pod_id)
        succeeded = self.terminal_status == "TERMINATED_HARVESTED"
        return {
            "status": self.terminal_status,
            "termination_reason": self.reason,
            "per_pod_404_observed": self.dual,
            "active_inventory_absent_observed": self.dual,
            "harvest": {"succeeded": succeeded},
        }


def _release(tmp_path: Path) -> tuple[Path, life.VerifiedRelease]:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / life.JOB_PATH).write_text("# fixed technical job\n")
    (repo / "contract.json").write_text("{}\n")
    receipt = tmp_path / "receipt.json"
    report = tmp_path / "import-report.json"
    receipt.write_text('{"receipt":"fixed"}\n')
    report.write_text('{"report":"fixed"}\n')
    release = life.VerifiedRelease(
        authorization_commit="a" * 40,
        manifest_path="release/stage-t.json",
        receipt_path=receipt,
        receipt_sha256=life.file_sha256(receipt),
        import_report_path=report,
        import_report_sha256=life.file_sha256(report),
        sync_paths=("contract.json", life.JOB_PATH),
    )
    return repo, release


def _run(tmp_path, *, provider_actions=("success",), admissions=(True,),
         sync_error=False, launch_error=False,
         terminal_status="TERMINATED_HARVESTED", reason="job_complete",
         dual=True, death=None, start_errors=(), terminal_advance_seconds=0,
         prior_stage_t_provider_seconds=0, provider_response_mutation=None):
    clock = Clock()
    log = []
    provider = FakeProvider(
        clock, provider_actions, log,
        response_mutation=provider_response_mutation)
    transport = FakeTransport(
        admissions, sync_error=sync_error, launch_error=launch_error, log=log)
    supervisor = FakeSupervisor(
        provider, terminal_status=terminal_status, reason=reason,
        dual=dual, death=death, start_errors=start_errors,
        terminal_advance_seconds=terminal_advance_seconds, log=log)
    repo, release = _release(tmp_path)
    runner = life.StageTLifecycle(
        repo=repo, session_root=tmp_path / "session", release=release,
        provider=provider, transport=transport, supervisor=supervisor,
        clock=clock, prior_stage_t_spend_usd="0",
        prior_stage_t_provider_seconds=prior_stage_t_provider_seconds,
        job_probe_command=["probe"], harvest_command=["harvest"],
    )
    return runner, provider, transport, supervisor, log


def test_happy_path_has_one_host_exact_sync_detach_harvest_delete_no_warm_hold(
        tmp_path):
    runner, provider, transport, supervisor, log = _run(tmp_path)
    result = runner.run()
    assert result["status"] == "COMPLETE"
    assert result["admitted_pod_id"] == "pod_stage_t_1"
    assert result["job_started"] is True
    assert provider.creates == 1 and provider.active == []
    assert transport.synced == ["contract.json", life.JOB_PATH]
    assert supervisor.harvests == 1
    assert result["events"][-1]["kind"] == "NO_WARM_HOLD_TERMINAL"
    admitted = next(row for row in result["events"]
                    if row["kind"] == "HOST_ADMITTED")
    assert admitted["evidence"]["driver_components"] == [580, 65, 6]
    create = next(row for row in log if isinstance(row, tuple)
                  and row[0] == "create")
    attempt = result["attempts"][0]
    assert create[2] == attempt["clock_started_epoch"]
    assert attempt["attempt_provider_seconds"] > 0
    assert result["observed_stage_t_provider_seconds"] == \
        attempt["attempt_provider_seconds"]
    assert result["events"][-1]["evidence"][
        "observed_stage_t_provider_seconds"] == \
        result["observed_stage_t_provider_seconds"]
    assert log.index(("watcher_start", "watchdog.json")) < \
        next(i for i, row in enumerate(log)
             if isinstance(row, tuple) and row[0] == "admit")


def test_one_no_capacity_retry_then_one_admitted_host(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("no_capacity", "success"))
    result = runner.run()
    assert provider.creates == 2
    assert [row["result"] for row in result["attempts"]] == [
        "NO_CAPACITY", "ALLOCATED"]
    assert result["admitted_pod_id"] == "pod_stage_t_2"


def test_two_no_capacity_responses_exhaust_without_allocation(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("no_capacity", "no_capacity"))
    result = runner.run()
    assert result["status"] == "NO_CAPACITY"
    assert provider.creates == 2 and result["admitted_pod_id"] is None


def test_ambiguous_allocation_is_fatal_and_never_retried(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("ambiguous", "success"))
    with pytest.raises(life.AllocationAmbiguous, match="retry forbidden"):
        runner.run()
    assert provider.creates == 1
    assert runner.state["status"] == "BLOCKED"


def test_admission_reject_is_positively_deleted_before_attempt_two(tmp_path):
    runner, provider, _transport, supervisor, log = _run(
        tmp_path, provider_actions=("success", "success"),
        admissions=(False, True))
    result = runner.run()
    assert provider.creates == 2 and supervisor.cleanup_requests == 1
    assert result["admitted_pod_id"] == "pod_stage_t_2"
    first_delete = log.index(("delete", "pod_stage_t_1"))
    second_create = log.index(("create", 2,
                               result["attempts"][1]["clock_started_epoch"]))
    assert first_delete < second_create
    assert Decimal(result["observed_stage_t_spend_usd"]) > 0
    first, second = result["attempts"]
    assert first["attempt_provider_seconds"] > 0
    assert first["admission_failure_stage"] == "REMOTE_HOST_ATTESTATION"
    assert first["admission_failure_code"] == \
        "REMOTE_HOST_ATTESTATION_REJECTED"
    assert first["host_attestation_received"] is False
    assert first["provider_allocation"]["machine_secure_cloud"] is True
    second_record = json.loads((
        runner.root / "attempt-2" / "watchdog.json").read_text())
    assert second_record["prior_stage_t_provider_seconds"] == \
        first["observed_stage_t_provider_seconds"]
    assert second_record["bounded_provider_seconds"] == \
        3300 - first["observed_stage_t_provider_seconds"]
    assert result["observed_stage_t_provider_seconds"] == \
        second["observed_stage_t_provider_seconds"]


def test_provider_schema_rejection_persists_safe_stage_code_and_shape(tmp_path):
    def wrong_sku(response):
        response["machine"]["gpuTypeId"] = "NVIDIA H100 80GB HBM3"
        response["env"] = ["DO_NOT_PERSIST_THIS_SECRET"]

    runner, provider, _transport, supervisor, _log = _run(
        tmp_path, provider_actions=("success", "success"),
        provider_response_mutation=wrong_sku)
    result = runner.run()
    assert result["status"] == "ADMISSION_REJECTED"
    assert provider.creates == 2 and supervisor.cleanup_requests == 2
    rejected = [
        event for event in result["events"]
        if event["kind"] == "ADMISSION_REJECTED_DELETED"]
    assert len(rejected) == 2
    for event in rejected:
        evidence = event["evidence"]
        assert evidence["admission_failure_stage"] == \
            "PROVIDER_ALLOCATION_SCHEMA"
        assert evidence["admission_failure_code"] == \
            "PROVIDER_MACHINE_GPU_SKU_MISMATCH"
        assert evidence["host_attestation_received"] is False
        assert evidence["provider_allocation"]["machine_gpu_type_id"] == \
            "NVIDIA H100 80GB HBM3"
        assert "env" not in evidence["provider_allocation"]
        assert "error_message" not in evidence
        assert "DO_NOT_PERSIST_THIS_SECRET" not in json.dumps(evidence)


def test_caller_prior_provider_seconds_reduce_first_attempt_window(tmp_path):
    runner, _provider, _transport, _supervisor, _log = _run(
        tmp_path, prior_stage_t_provider_seconds=900)
    result = runner.run()
    record = json.loads((
        runner.root / "attempt-1" / "watchdog.json").read_text())
    assert record["prior_stage_t_provider_seconds"] == 900
    assert record["bounded_provider_seconds"] == 2400
    assert result["observed_stage_t_provider_seconds"] > 900


def test_caller_prior_seconds_without_cleanup_lead_forbid_allocation(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, prior_stage_t_provider_seconds=3180)
    with pytest.raises(life.V13LifecycleError, match="cannot fund safe cleanup"):
        runner.run()
    assert provider.creates == 0
    assert runner.state["status"] == "BLOCKED"


def test_pre_watchdog_rejection_charges_seconds_before_second_attempt(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("success", "success"),
        start_errors=(True, False))
    result = runner.run()
    assert provider.creates == 2
    first = result["attempts"][0]
    assert first["result"] == "PRE_WATCHDOG_REJECTED_DELETED"
    assert first["attempt_provider_seconds"] > 0
    second_record = json.loads((
        runner.root / "attempt-2" / "watchdog.json").read_text())
    assert second_record["prior_stage_t_provider_seconds"] == \
        first["observed_stage_t_provider_seconds"]
    assert second_record["bounded_provider_seconds"] == \
        3300 - first["observed_stage_t_provider_seconds"]


def test_insufficient_cumulative_seconds_forbid_second_allocation(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("success", "success"),
        admissions=(False, True), terminal_advance_seconds=3180)
    with pytest.raises(life.V13LifecycleError, match="cannot fund safe cleanup"):
        runner.run()
    assert provider.creates == 1
    assert runner.state["status"] == "BLOCKED"
    assert runner.state["observed_stage_t_provider_seconds"] > 3180
    assert runner.state["events"][-1]["kind"] == \
        "PROVIDER_SECONDS_EXHAUSTED"


def test_admission_cleanup_without_dual_deletion_forbids_attempt_two(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, provider_actions=("success", "success"),
        admissions=(False, True), dual=False)
    with pytest.raises(life.V13LifecycleError, match="dual deletion"):
        runner.run()
    assert provider.creates == 1


@pytest.mark.parametrize("phase", ["sync", "launch"])
def test_no_retry_after_admission_or_job_start(tmp_path, phase):
    runner, provider, _transport, supervisor, _log = _run(
        tmp_path, provider_actions=("success", "success"),
        sync_error=phase == "sync", launch_error=phase == "launch")
    with pytest.raises(life.V13LifecycleError, match="retry forbidden"):
        runner.run()
    assert provider.creates == 1 and provider.active == []
    assert supervisor.cleanup_requests == 1
    assert runner.state["attempts"][0]["attempt_provider_seconds"] > 0
    assert runner.state["observed_stage_t_provider_seconds"] == \
        runner.state["attempts"][0]["observed_stage_t_provider_seconds"]


@pytest.mark.parametrize("death", ["before_harvest", "after_harvest"])
def test_unexpected_watcher_death_invokes_emergency_and_harvests_once(
        tmp_path, death):
    runner, provider, _transport, supervisor, log = _run(
        tmp_path, death=death)
    result = runner.run()
    assert result["status"] == "COMPLETE" and provider.active == []
    assert supervisor.harvests == 1
    assert ("emergency_cleanup", death) in log


def test_job_death_and_harvest_failure_still_delete(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, terminal_status="TERMINATED_HARVEST_FAILED",
        reason="process_death")
    result = runner.run()
    assert result["status"] == "HARVEST_FAILED"
    assert provider.active == []
    assert result["events"][-1]["evidence"]["termination_reason"] == \
        "process_death"


@pytest.mark.parametrize("reason", [
    "deadline", "rate_increase", "provider_identity", "process_death",
])
def test_non_job_terminal_harvest_is_never_false_complete(tmp_path, reason):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, terminal_status="TERMINATED_HARVESTED", reason=reason)
    result = runner.run()
    assert result["status"] == "STOPPED"
    assert provider.active == []


def test_job_terminal_requires_dual_deletion(tmp_path):
    runner, provider, _transport, _supervisor, _log = _run(
        tmp_path, dual=False)
    with pytest.raises(life.V13LifecycleError, match="retry forbidden"):
        runner.run()
    assert provider.creates == 1


def test_fixed_job_probe_exit_contract():
    for status in ("ALLOCATING", "WATCHDOG_STARTED", "ADMITTED", "SYNCED"):
        assert life.job_probe_exit(lifecycle_status=status, remote_state=None) == 0
    assert life.job_probe_exit(
        lifecycle_status="FORCE_CLEANUP", remote_state=None) == 21
    assert life.job_probe_exit(
        lifecycle_status="JOB_STARTED", remote_state="RUNNING") == 0
    assert life.job_probe_exit(
        lifecycle_status="JOB_STARTED", remote_state="COMPLETE") == 20
    assert life.job_probe_exit(
        lifecycle_status="JOB_STARTED", remote_state="DEAD") == 21


def test_run_cli_requires_exact_prior_provider_seconds_input():
    spec = importlib.util.spec_from_file_location("stage_t_lifecycle_cli", CLI)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    argv = [
        "run", "--repo", ".", "--authorization-commit", "a" * 40,
        "--manifest", "manifest.json", "--receipt", "receipt.json",
        "--receipt-sha256", "b" * 64, "--import-report", "report.json",
        "--import-report-sha256", "c" * 64, "--session-root", "session",
        "--prior-stage-t-spend-usd", "0", "--ssh-key", "ssh-key",
        "--hf-token", "hf-token", "--primary-batch-id", "batch",
    ]
    with pytest.raises(SystemExit) as missing:
        cli.parser().parse_args(argv)
    assert missing.value.code == 2
    parsed = cli.parser().parse_args([
        *argv, "--prior-stage-t-provider-seconds", "0"])
    assert parsed.prior_stage_t_provider_seconds == 0


def test_harvest_manifest_hashes_every_required_category(tmp_path):
    root = tmp_path / "harvest"
    for category in life.HARVEST_ROOTS:
        path = root / category / f"{category}.txt"
        path.parent.mkdir(parents=True)
        path.write_text(category)
    manifest = life.build_harvest_manifest(root)
    assert manifest["file_count"] == len(life.HARVEST_ROOTS)
    assert {row["category"] for row in manifest["files"]} == \
        set(life.HARVEST_ROOTS)
    assert all(len(row["sha256"]) == 64 for row in manifest["files"])


def _binding_fixture(tmp_path):
    repo = tmp_path / "release-repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "release").mkdir()
    contents = {
        life.JOB_PATH: "import helper\n",
        "src/helper.py": "VALUE = 1\n",
        "PREREG.md": "PREREG.md\n",
    }
    for relative, content in contents.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    commit = "a" * 40
    manifest_path = "release/stage-t.json"
    report_relative = "release/import-audit.json"
    inventory_paths = sorted((
        "PREREG.md", "contract.json", report_relative, life.JOB_PATH,
        "src/helper.py",
    ))
    contract_doc = {
        "schema": life.CONTRACT_SCHEMA,
        "design_id": life.DESIGN_ID,
        "stage": "TECHNICAL_CANARY",
        "inventory_paths": inventory_paths,
    }
    (repo / "contract.json").write_bytes(
        life.canonical_json_bytes(contract_doc) + b"\n")
    report = repo / report_relative
    report.write_text("placeholder\n")
    report_doc = life.audit_stage_t_imports(
        repo, contract_path=Path("contract.json"),
        execution_roots=[life.JOB_PATH])
    report.write_bytes(life.canonical_json_bytes(report_doc) + b"\n")
    inventory = []
    for relative in inventory_paths:
        path = repo / relative
        inventory.append({
            "path": relative, "mode": "100644",
            "bytes": path.stat().st_size, "sha256": life.file_sha256(path),
        })
    manifest = {"inventory": inventory}
    (repo / manifest_path).write_bytes(
        life.canonical_json_bytes(manifest) + b"\n")
    receipt = tmp_path / "receipts" / "powered-v13-stage-t-launch-receipt.json"
    receipt.parent.mkdir()
    receipt_doc = {"design_id": life.DESIGN_ID, "stage": "TECHNICAL_CANARY",
                   "authorization_commit": commit, "manifest_path": manifest_path}
    receipt.write_bytes(life.canonical_json_bytes(receipt_doc) + b"\n")
    receipt_sha = life.file_sha256(receipt)
    report_sha = life.file_sha256(report)

    def verify_checkout(**_kwargs):
        return {"status": "PASS", "detached_head": commit,
                "authorization": {
                    "authorization_commit": commit,
                    "manifest_sha256": life.file_sha256(repo / manifest_path),
                    "changed_paths": ["PREREG.md", manifest_path],
                }, "receipt": {"receipt_sha256": receipt_sha}}
    return (repo, commit, manifest_path, receipt, receipt_sha, report,
            report_sha, verify_checkout)


def test_release_binding_derives_exact_authorized_sync_allowlist(tmp_path):
    args = _binding_fixture(tmp_path)
    verified = life._verify_release_binding_core(
        repo=args[0], expected_authorization_commit=args[1],
        manifest_path=args[2], receipt_path=args[3],
        expected_receipt_sha256=args[4], import_report_path=args[5],
        expected_import_report_sha256=args[6], verify_checkout=args[7])
    assert verified.sync_paths == tuple(sorted((
        "PREREG.md", "contract.json", "release/import-audit.json",
        "release/stage-t.json", life.JOB_PATH, "src/helper.py")))


def test_release_binding_public_and_remote_setup_wrappers_are_fixed() -> None:
    public = inspect.signature(life.verify_release_binding).parameters
    setup = inspect.signature(
        life._verify_remote_setup_release_binding).parameters
    assert "verify_checkout" not in public
    assert "verify_checkout" not in setup
    assert "_verify_remote_setup_release_binding" not in life.__all__
    assert "verify_checkout=verify_stage_t_checkout" in inspect.getsource(
        life.verify_release_binding)
    assert "verify_checkout=_verify_stage_t_remote_setup_checkout" in \
        inspect.getsource(life._verify_remote_setup_release_binding)


def test_release_binding_rejects_wrong_expected_receipt_hash_before_checkout(
        tmp_path):
    args = _binding_fixture(tmp_path)

    def forbidden_checkout(**_kwargs):
        pytest.fail("checkout ran after receipt hash rejection")

    with pytest.raises(life.V13LifecycleError, match="receipt hash differs"):
        life._verify_release_binding_core(
            repo=args[0], expected_authorization_commit=args[1],
            manifest_path=args[2], receipt_path=args[3],
            expected_receipt_sha256="0" * 64,
            import_report_path=args[5],
            expected_import_report_sha256=args[6],
            verify_checkout=forbidden_checkout)


def test_release_binding_rejects_setup_receipt_binding_before_checkout(
        tmp_path):
    args = list(_binding_fixture(tmp_path))
    receipt = json.loads(args[3].read_text())
    receipt["manifest_path"] = "release/other.json"
    args[3].write_bytes(life.canonical_json_bytes(receipt) + b"\n")
    args[4] = life.file_sha256(args[3])

    def forbidden_checkout(**_kwargs):
        pytest.fail("checkout ran after outer receipt binding rejection")

    with pytest.raises(life.V13LifecycleError, match="receipt binding differs"):
        life._verify_release_binding_core(
            repo=args[0], expected_authorization_commit=args[1],
            manifest_path=args[2], receipt_path=args[3],
            expected_receipt_sha256=args[4], import_report_path=args[5],
            expected_import_report_sha256=args[6],
            verify_checkout=forbidden_checkout)


def test_release_binding_rejects_caller_pinned_forged_minimal_pass_report(tmp_path):
    args = list(_binding_fixture(tmp_path))
    report = args[5]
    forged = {
        "schema": life.IMPORT_REPORT_SCHEMA, "design_id": life.DESIGN_ID,
        "stage": "TECHNICAL_CANARY", "status": "PASS", "semantic_n": 0,
        "production_entropy_requested": False,
        "execution_roots": [life.JOB_PATH], "contract_path": "contract.json",
    }
    forged["report_sha256"] = life.sha256_bytes(
        life.canonical_json_bytes(forged))
    report.write_bytes(life.canonical_json_bytes(forged) + b"\n")
    args[6] = life.file_sha256(report)
    manifest_path = args[0] / args[2]
    manifest = json.loads(manifest_path.read_text())
    row = next(row for row in manifest["inventory"]
               if row["path"] == "release/import-audit.json")
    row["bytes"] = report.stat().st_size
    row["sha256"] = args[6]
    manifest_path.write_bytes(life.canonical_json_bytes(manifest) + b"\n")

    def verify_checkout(**_kwargs):
        evidence = args[7](**_kwargs)
        evidence["authorization"]["manifest_sha256"] = life.file_sha256(
            manifest_path)
        return evidence

    with pytest.raises(life.V13LifecycleError, match="recomputation"):
        life._verify_release_binding_core(
            repo=args[0], expected_authorization_commit=args[1],
            manifest_path=args[2], receipt_path=args[3],
            expected_receipt_sha256=args[4], import_report_path=args[5],
            expected_import_report_sha256=args[6],
            verify_checkout=verify_checkout)


def test_release_binding_rejects_production_path(tmp_path):
    args = list(_binding_fixture(tmp_path))
    manifest_path = args[0] / args[2]
    manifest = json.loads(manifest_path.read_text())
    forbidden = sorted(life.FORBIDDEN_INVENTORY_PATHS)[0]
    path = args[0] / forbidden
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("forbidden\n")
    manifest["inventory"].append({
        "path": forbidden, "mode": "100644", "bytes": path.stat().st_size,
        "sha256": life.file_sha256(path),
    })
    manifest["inventory"] = sorted(manifest["inventory"], key=lambda row: row["path"])
    manifest_path.write_bytes(life.canonical_json_bytes(manifest) + b"\n")

    def verify_checkout(**_kwargs):
        evidence = args[7](**_kwargs)
        evidence["authorization"]["manifest_sha256"] = life.file_sha256(manifest_path)
        return evidence
    with pytest.raises(life.V13LifecycleError, match="production-only"):
        life._verify_release_binding_core(
            repo=args[0], expected_authorization_commit=args[1],
            manifest_path=args[2], receipt_path=args[3],
            expected_receipt_sha256=args[4], import_report_path=args[5],
            expected_import_report_sha256=args[6], verify_checkout=verify_checkout)


def test_provider_snapshot_is_safe_exact_and_secret_free():
    result = life.validate_provider_snapshot({
        "balance_usd": "57.12", "spend_limit_usd": "80",
        "active_pod_ids": ["pod_1"],
    })
    assert set(result) == {"balance_usd", "spend_limit_usd", "active_pod_ids"}
    with pytest.raises(life.V13LifecycleError, match="fields differ"):
        life.validate_provider_snapshot({**result, "api_key": "secret"})


def _valid_admission_evidence():
    return {
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "memory_mib": 81920,
        "cuda_available": True,
        "gpu_count": 1,
        "torch_gpu_name": "NVIDIA A100 80GB PCIe",
        "gpu_uuid": "GPU-fixed",
        "driver_version": "580.159.03",
    }


def test_admission_accepts_preserved_nested_secure_allocation_shape(
        preserved_create_response):
    pod = preserved_create_response
    assert "cloudType" not in pod
    observed = life.validate_admission(pod, _valid_admission_evidence())
    assert observed["driver_components"] == [580, 159, 3]
    assert observed["provider_allocation"] == {
        "schema": "runpod-secure-allocation-shape-v1",
        "machine_present": True,
        "machine_secure_cloud": True,
        "machine_gpu_type_id": "NVIDIA A100 80GB PCIe",
        "gpu_count": 1,
        "top_level_cloud_type_present": False,
        "top_level_cloud_type": None,
    }

    with_top_level = deepcopy(pod)
    with_top_level["cloudType"] = "SECURE"
    assert life.validate_admission(
        with_top_level, _valid_admission_evidence())["provider_allocation"][
            "top_level_cloud_type"] == "SECURE"


@pytest.mark.parametrize(("mutation", "code"), [
    (lambda pod: pod.pop("machine"), "PROVIDER_MACHINE_METADATA_ABSENT"),
    (lambda pod: pod["machine"].pop("secureCloud"),
     "PROVIDER_MACHINE_NOT_SECURE"),
    (lambda pod: pod["machine"].update({"secureCloud": False}),
     "PROVIDER_MACHINE_NOT_SECURE"),
    (lambda pod: pod["machine"].update({"gpuTypeId": "NVIDIA H100 80GB HBM3"}),
     "PROVIDER_MACHINE_GPU_SKU_MISMATCH"),
    (lambda pod: pod.update({"gpuCount": 2}),
     "PROVIDER_GPU_COUNT_MISMATCH"),
    (lambda pod: pod.update({"cloudType": "COMMUNITY"}),
     "PROVIDER_TOP_LEVEL_CLOUD_TYPE_MISMATCH"),
])
def test_admission_rejects_missing_false_or_wrong_nested_provider_shape(
        preserved_create_response, mutation, code):
    mutation(preserved_create_response)
    with pytest.raises(life.AdmissionRejected) as caught:
        life.validate_admission(
            preserved_create_response, _valid_admission_evidence())
    assert caught.value.stage == "PROVIDER_ALLOCATION_SCHEMA"
    assert caught.value.code == code
    assert caught.value.evidence["schema"] == \
        "runpod-secure-allocation-shape-v1"


@pytest.mark.parametrize("changes, message", [
    ({"gpu_name": "NVIDIA H100 80GB HBM3"}, "GPU name differs"),
    ({"memory_mib": 40960}, "GPU memory differs"),
    ({"driver_version": "550.90.12"}, "below 580.65.06"),
    ({"driver_version": "not-a-driver"}, "version is malformed"),
    ({"gpu_count": 2}, "one-GPU/CUDA/UUID"),
    ({"cuda_available": False}, "one-GPU/CUDA/UUID"),
])
def test_admission_rejects_wrong_paid_host(changes, message):
    pod = _provider_response("pod_stage_t_1")
    evidence = {**_valid_admission_evidence(), **changes}
    with pytest.raises(life.V13LifecycleError, match=message):
        life.validate_admission(pod, evidence)


class FakePodModule:
    DEFAULT_GPU = "NVIDIA A100 80GB PCIe"

    def __init__(self, state_path, *, create_error=None):
        self.state_path = state_path
        self.create_error = create_error
        self.calls = []

    def gql(self, query):
        self.calls.append(("gql", query))
        return {"data": {"myself": {"clientBalance": 57.12,
                                     "spendLimit": 80}}}

    def api(self, method, path):
        self.calls.append((method, path))
        if method == "GET" and path == "/pods":
            return []
        if method == "GET":
            raise urllib.error.HTTPError("x", 404, "gone", {}, None)
        return {}

    def create(self, gpu):
        self.calls.append(("create", gpu))
        if self.create_error:
            raise self.create_error
        value = _provider_response("pod_stage_t_1")
        self.state_path.write_text(json.dumps(value))
        return value


def test_runpod_adapter_reuses_pod_api_and_classifies_capacity(tmp_path):
    state = tmp_path / "pod.json"
    module = FakePodModule(state)
    provider = life.RunPodProvider(state_path=state, module=module)
    assert os.environ["SC_POD_ALLOWED_CUDA"] == "13.0"
    assert os.environ["SC_POD_CLOUD"] == "SECURE"
    assert "SC_POD_SPOT" not in os.environ
    assert provider.safe_snapshot() == {
        "balance_usd": "57.12", "spend_limit_usd": "80",
        "active_pod_ids": [],
    }
    os.environ["SC_POD_ALLOWED_CUDA"] = ""
    allocation = provider.create_secure_a100()
    assert allocation.pod_id == "pod_stage_t_1"
    assert os.environ["SC_POD_ALLOWED_CUDA"] == "13.0"
    assert ("create", "NVIDIA A100 80GB PCIe") in module.calls
    with pytest.raises(life.PodNotFound):
        provider.get_pod("pod_stage_t_1")

    no_capacity = FakePodModule(
        state, create_error=urllib.error.HTTPError(
            "x", 500, "none", {}, None))
    provider = life.RunPodProvider(state_path=state, module=no_capacity)
    provider.safe_snapshot()
    with pytest.raises(life.NoCapacity):
        provider.create_secure_a100()


class AmbiguousCreatePodModule:
    def __init__(self, *, lose_state=True, raise_after_create=False):
        self.active = []
        self.lose_state = lose_state
        self.raise_after_create = raise_after_create
        self.state_path = None
        self.deleted = []

    def gql(self, _query):
        return {"data": {"myself": {"clientBalance": 57.12,
                                     "spendLimit": 80}}}

    def api(self, method, path):
        if method == "GET" and path == "/pods":
            return list(self.active)
        pod_id = path.rsplit("/", 1)[-1]
        match = next((pod for pod in self.active if pod["id"] == pod_id), None)
        if method == "GET":
            if match is None:
                raise urllib.error.HTTPError("x", 404, "gone", {}, None)
            return dict(match)
        if method == "DELETE":
            self.deleted.append(pod_id)
            self.active = [pod for pod in self.active if pod["id"] != pod_id]
            return {}
        raise AssertionError((method, path))

    def create(self, gpu):
        assert gpu == "NVIDIA A100 80GB PCIe"
        pod = _provider_response(
            "pod_recovered", name=os.environ["SC_POD_NAME"])
        self.active.append(pod)
        if self.raise_after_create:
            raise OSError("response lost after POST")
        if not self.lose_state:
            self.state_path.write_text(json.dumps(pod))
        return dict(pod)


@pytest.mark.parametrize("raise_after_create", [True, False])
def test_runpod_adapter_reconciles_and_deletes_attributable_ambiguous_create(
        tmp_path, raise_after_create):
    state = tmp_path / "pod.json"
    module = AmbiguousCreatePodModule(
        lose_state=True, raise_after_create=raise_after_create)
    module.state_path = state
    provider = life.RunPodProvider(state_path=state, module=module)
    assert provider.safe_snapshot()["active_pod_ids"] == []
    with pytest.raises(life.AllocationAmbiguous, match="retry forbidden"):
        provider.create_secure_a100()
    assert module.deleted == ["pod_recovered"] and module.active == []
    evidence_paths = list(tmp_path.glob("pod.json.attempt-*.json"))
    assert len(evidence_paths) == 1
    evidence = json.loads(evidence_paths[0].read_text())
    assert evidence["status"] == "ATTRIBUTED_DELETED"
    assert evidence["per_pod_404_observed"] is True
    assert evidence["active_inventory_absent_observed"] is True
    inspection = evidence["new_pod_inspections"][0]
    assert inspection["secure_one_gpu"] is True
    assert inspection["provider_allocation"]["machine_secure_cloud"] is True
    assert inspection["provider_allocation"]["machine_gpu_type_id"] == \
        "NVIDIA A100 80GB PCIe"
    assert inspection["provider_allocation"][
        "top_level_cloud_type_present"] is False


def test_runpod_adapter_zero_new_pods_is_fatal_ambiguous_without_retry(tmp_path):
    state = tmp_path / "pod.json"

    class Module(AmbiguousCreatePodModule):
        def create(self, _gpu):
            raise OSError("connection lost")

    module = Module()
    now = [0.0]
    provider = life.RunPodProvider(
        state_path=state, module=module, reconcile_wait_seconds=3,
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(
            0, now[0] + max(seconds, 1)))
    provider.safe_snapshot()
    with pytest.raises(life.AllocationAmbiguous, match="retry forbidden"):
        provider.create_secure_a100()
    assert module.deleted == []
    evidence = json.loads(next(tmp_path.glob(
        "pod.json.attempt-*.json")).read_text())
    assert evidence["status"] == "NO_NEW_POD_AMBIGUOUS"


def test_runpod_adapter_polls_eventual_inventory_then_deletes_nonce_match(tmp_path):
    state = tmp_path / "pod.json"
    module = AmbiguousCreatePodModule(raise_after_create=True)
    module.state_path = state
    now = [0.0]

    def sleep(seconds):
        now[0] += max(seconds, 1)
        if not module.active:
            module.active.append(_provider_response(
                "pod_eventual", name=os.environ["SC_POD_NAME"]))

    # The module's create adds immediately; override it to model eventual
    # provider visibility after the POST-side exception.
    module.create = lambda _gpu: (_ for _ in ()).throw(OSError("timeout"))
    provider = life.RunPodProvider(
        state_path=state, module=module, reconcile_wait_seconds=5,
        monotonic=lambda: now[0], sleep=sleep)
    provider.safe_snapshot()
    with pytest.raises(life.AllocationAmbiguous, match="retry forbidden"):
        provider.create_secure_a100()
    assert module.deleted == ["pod_eventual"]
    evidence = json.loads(next(tmp_path.glob(
        "pod.json.attempt-*.json")).read_text())
    assert evidence["inventory_poll_count"] >= 2
    assert evidence["status"] == "ATTRIBUTED_DELETED"


def test_runpod_adapter_deletes_all_nonce_matches_but_not_concurrent_pod(tmp_path):
    state = tmp_path / "pod.json"
    module = AmbiguousCreatePodModule(raise_after_create=True)

    def create(_gpu):
        name = os.environ["SC_POD_NAME"]
        module.active.extend([
            _provider_response("pod_owned_1", name=name),
            _provider_response("pod_owned_2", name=name),
            _provider_response("pod_other", name="someone-else"),
        ])
        raise OSError("response lost")

    module.create = create
    provider = life.RunPodProvider(state_path=state, module=module)
    provider.safe_snapshot()
    with pytest.raises(life.AllocationAmbiguous, match="retry forbidden"):
        provider.create_secure_a100()
    assert module.deleted == ["pod_owned_1", "pod_owned_2"]
    assert [pod["id"] for pod in module.active] == ["pod_other"]
    evidence = json.loads(next(tmp_path.glob(
        "pod.json.attempt-*.json")).read_text())
    assert evidence["attributed_pod_ids"] == ["pod_owned_1", "pod_owned_2"]
    assert evidence["status"] == "ATTRIBUTED_DELETED"


def test_concrete_ssh_transport_admits_before_exact_sync_bootstrap_and_detach(
        tmp_path):
    key = tmp_path / "ssh-key"
    token = tmp_path / "hf-token"
    key.write_text("key\n")
    token.write_text("token\n")
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / life.JOB_PATH).write_text("runner\n")
    (repo / "contract.json").write_text("{}\n")
    receipt = tmp_path / "receipt.json"
    report = repo / "import-report.json"
    receipt.write_text("{}\n")
    report.write_text("{}\n")
    commands = []

    class Provider:
        def get_pod(self, pod_id):
            assert pod_id == "pod_stage_t_1"
            return {"id": pod_id, "publicIp": "203.0.113.8",
                    "portMappings": {"22": "2222"}}

    def run(command, **kwargs):
        commands.append((list(command), kwargs.get("cwd")))
        text = command[-1] if command and command[0] == "ssh" else ""
        if "nvidia-smi --query-gpu" in text:
            stdout = (
                "NVIDIA A100 80GB PCIe, GPU-fixed, 580.159.03, 81920\n")
        elif "create_stage_t_launch_receipt" in text:
            stdout = json.dumps({
                "fresh_receipt_sha256": "d" * 64,
                "fresh_receipt_created_utc":
                    "2026-07-12T22:00:00.000000Z",
            }, sort_keys=True, separators=(",", ":")) + "\n"
        elif "torch.cuda.is_available" in text:
            stdout = "NVIDIA A100 80GB PCIe\n"
        elif "_verify-remote-setup" in text:
            stdout = '{"status": "PASS"}\n'
        elif "nohup /bin/bash" in text:
            stdout = "4321\n"
        elif "for i in $(seq 1 30)" in text:
            stdout = "PASS\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    transport = life.OpenSshTransport(
        provider=Provider(), ssh_key=key, hf_token=token,
        authorization_commit="a" * 40, primary_batch_id="stage-t-batch-1",
        manifest_path="release/stage-t.json",
        receipt_sha256=life.file_sha256(receipt),
        import_report_path="import-report.json",
        import_report_sha256=life.file_sha256(report),
        run_command=run, sleep=lambda _seconds: None)
    allocation = life.Allocation(
        response=_provider_response("pod_stage_t_1"), state_bytes=b"{}")
    admission = transport.admit(allocation, timeout_seconds=420)
    assert life.validate_admission(allocation.response, admission)[
        "driver_components"] == [580, 159, 3]
    assert not any("pip install" in " ".join(command)
                   for command, _cwd in commands)
    synced = transport.sync(
        allocation, repo=repo,
        relative_paths=("contract.json", "import-report.json", life.JOB_PATH),
        external_files=(receipt, report), timeout_seconds=300)
    assert synced["detached_head"] == "a" * 40
    assert synced["model_download_auth_mode"] == "token_file"
    assert synced["release_receipt_sha256"] == life.file_sha256(receipt)
    assert synced["fresh_launch_receipt_sha256"] == "d" * 64
    assert synced["fresh_launch_receipt_created_utc"] == \
        "2026-07-12T22:00:00.000000Z"
    launched = transport.launch_detached(
        allocation, job_path=life.JOB_PATH,
        authorization_commit="a" * 40, timeout_seconds=60)
    assert launched["pipe_eof"] is launched["child_alive"] is True
    launch_remote = next(
        command[-1] for command, _cwd in commands
        if command[0] == "ssh" and "nohup /bin/bash" in command[-1])
    syntax = subprocess.run(
        ["bash", "-n", "-c", launch_remote], text=True,
        capture_output=True, check=False)
    assert syntax.returncode == 0, syntax.stderr
    rendered = "\n".join(" ".join(command) for command, _cwd in commands)
    assert "git clone --quiet --no-checkout --sparse" in rendered
    assert "--filter=blob:none" not in rendered
    assert "uv==0.9.18" in rendered and "uv python install 3.12.11" in rendered
    assert "torch==2.12.1" in rendered and "transformers==5.0.0" in rendered
    assert "2.12.1+cu130" in rendered and "torch.version.cuda" in rendered
    assert "huggingface-hub==1.22.0" in rendered
    assert "--primary-batch-id stage-t-batch-1" in rendered
    assert "--output-parent /workspace/powered-v13-stage-t/artifacts/results" in rendered
    assert "--receipt-directory /workspace/powered-v13-stage-t/launch-receipts" \
        in rendered
    assert rendered.index("_verify-remote-setup") < rendered.index("pip install")
    assert not any(flag in rendered for flag in (
        "--allow-stale", "--skip-age", "--max-age", "--age-policy"))
    assert rendered.index("snapshot_download") < rendered.index(
        "create_stage_t_launch_receipt") < rendered.index("nohup /bin/bash")
    assert "RUN_COMPLETE.json" in rendered and "WORKER_EXITED_ZERO" in rendered
    assert rendered.index("write(terminal") < rendered.index(
        "write(state_path")
    assert "/external/import-report.json" not in rendered
    assert "/external/hf-token" not in rendered
    assert "/secrets/hf-token" in rendered
    assert "/launch-receipts/powered-v13-technical-canary-launch-receipt.json" \
        in rendered
    assert "/receipts/subject-launch-receipt.json" in rendered
    assert "powered_v13_recipe" not in rendered


def test_cli_keeps_normal_preallocation_and_private_remote_setup_paths_fixed():
    spec = importlib.util.spec_from_file_location("stage_t_lifecycle_cli", CLI)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run_source = inspect.getsource(cli.command_run)
    setup_source = inspect.getsource(cli._command_verify_remote_setup)
    assert run_source.index("verify_release_binding") < run_source.index(
        "RunPodProvider")
    assert "_verify_remote_setup_release_binding" not in run_source
    assert "_verify_remote_setup_release_binding" in setup_source
    assert "verify_release_binding(" not in setup_source

    parsed = cli.parser()
    subparsers = next(
        action for action in parsed._actions
        if isinstance(action, argparse._SubParsersAction))
    for name in ("run", "verify-release", "_verify-remote-setup"):
        options = {
            option
            for action in subparsers.choices[name]._actions
            for option in action.option_strings
        }
        assert not ({
            "--allow-stale", "--skip-age", "--max-age", "--age-policy",
            "--verify-checkout",
        } & options)


@pytest.mark.parametrize("status,returncode", [("COMPLETE", 0), ("DEAD", 22)])
def test_terminal_publisher_fsync_path_writes_receipt_before_state(
        tmp_path, status, returncode):
    receipt = tmp_path / "receipts/job-terminal.json"
    state = tmp_path / "partials/job-state.json"
    receipt.parent.mkdir()
    state.parent.mkdir()
    state.write_text('{"state":"RUNNING"}\n')
    result = subprocess.run([
        sys.executable, "-c", life.JOB_TERMINAL_PUBLISHER,
        str(receipt), str(state), str(returncode), status, "batch-1",
    ], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(receipt.read_text()) == {
        "schema": life.JOB_TERMINAL_SCHEMA,
        "status": status, "returncode": returncode,
        "primary_batch_id": "batch-1",
    }
    assert json.loads(state.read_text()) == {
        "state": status, "returncode": returncode,
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_watchdog_harvest_rejects_missing_terminal_receipt(
        tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("stage_t_harvest_cli", CLI)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    pod_state = tmp_path / "pod-state.json"
    key = tmp_path / "ssh-key"
    pod_state.write_text('{"id":"pod_stage_t_1"}\n')
    key.write_text("key\n")

    class Provider:
        pass

    monkeypatch.setattr(cli, "RunPodProvider", lambda **_kwargs: Provider())
    monkeypatch.setattr(
        cli, "_endpoint", lambda _provider, _state: ("203.0.113.8", 2222))

    def run(command, **_kwargs):
        destination = Path(command[-1])
        destination.mkdir(parents=True, exist_ok=True)
        source = command[-2]
        category = source.rstrip("/").rsplit("/", 1)[-1]
        name = "job-start.json" if category == "receipts" else "one.json"
        (destination / name).write_text("{}\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cli.subprocess, "run", run)
    destination = tmp_path / "harvest"
    output = tmp_path / "harvest-manifest.json"
    with pytest.raises(life.V13LifecycleError, match="terminal receipt"):
        cli.command_watchdog_harvest(argparse.Namespace(
            pod_state=pod_state, ssh_key=key, destination=destination,
            output=output, primary_batch_id="batch-1"))
    assert not destination.exists()
    assert not output.exists()


def test_concrete_ssh_transport_bounds_endpoint_wait_without_commands(tmp_path):
    key = tmp_path / "ssh-key"
    token = tmp_path / "hf-token"
    key.write_text("key\n")
    token.write_text("token\n")
    now = [0.0]

    class Provider:
        def get_pod(self, pod_id):
            return {"id": pod_id, "publicIp": None, "portMappings": {}}

    transport = life.OpenSshTransport(
        provider=Provider(), ssh_key=key, hf_token=token,
        authorization_commit="a" * 40, primary_batch_id="batch",
        manifest_path="release/stage-t.json", receipt_sha256="b" * 64,
        import_report_path="import-report.json",
        import_report_sha256="c" * 64,
        run_command=lambda *_args, **_kwargs: pytest.fail("command ran"),
        monotonic=lambda: now[0], sleep=lambda seconds: now.__setitem__(
            0, now[0] + max(seconds, 1)))
    allocation = life.Allocation(response={"id": "pod_stage_t_1"},
                                 state_bytes=b"{}")
    with pytest.raises(life.AdmissionRejected, match="endpoint wait"):
        transport.admit(allocation, timeout_seconds=5)


def test_concrete_ssh_admission_rejects_nvidia_smi_host_when_torch_cuda_fails(
        tmp_path):
    key = tmp_path / "ssh-key"
    token = tmp_path / "hf-token"
    key.write_text("key\n")
    token.write_text("token\n")

    class Provider:
        def get_pod(self, pod_id):
            return {"id": pod_id, "publicIp": "203.0.113.8",
                    "portMappings": {"22": "2222"}}

    def run(command, **_kwargs):
        remote = command[-1]
        if "nvidia-smi --query-gpu" in remote:
            return subprocess.CompletedProcess(
                command, 0,
                "NVIDIA A100 80GB PCIe, GPU-fixed, 580.159.03, 81920\n", "")
        assert "torch.cuda.is_available" in remote
        return subprocess.CompletedProcess(command, 1, "", "CUDA unavailable")

    transport = life.OpenSshTransport(
        provider=Provider(), ssh_key=key, hf_token=token,
        authorization_commit="a" * 40, primary_batch_id="batch",
        manifest_path="release/stage-t.json", receipt_sha256="b" * 64,
        import_report_path="import-report.json",
        import_report_sha256="c" * 64, run_command=run)
    allocation = life.Allocation(response={"id": "pod_stage_t_1"},
                                 state_bytes=b"{}")
    with pytest.raises(life.V13LifecycleError, match="bounded command failed"):
        transport.admit(allocation, timeout_seconds=420)


def _watchdog_fixture(tmp_path):
    pod_state = tmp_path / "pod-state.json"
    pod_state.write_bytes(b'{"costPerHr":"1","id":"pod_stage_t_1"}\n')
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_bytes(b'{"force_cleanup_reason":null,"status":"WATCHDOG_STARTED"}\n')
    record_path = tmp_path / "watchdog.json"
    record = life.build_record(
        pod_id="pod_stage_t_1", created_cost_per_hr_usd="1",
        prior_stage_t_spend_usd="0", prior_stage_t_provider_seconds=0,
        provider_clock_started_epoch=1_000_000,
        pod_state_sha256=life.file_sha256(pod_state),
        create_response_sha256="a" * 64, job_sha256="b" * 64,
        release_receipt_sha256="c" * 64,
        job_probe_command=["probe"], harvest_command=["harvest"])
    life.write_record_exclusive(record_path, record)
    return record_path, pod_state, lifecycle


def test_launchd_backend_submits_caffeinated_watch_and_verifies_running(tmp_path):
    record, pod_state, lifecycle = _watchdog_fixture(tmp_path)
    active = set()
    commands = []

    def run(command, **_kwargs):
        commands.append(list(command))
        if command[1] == "print":
            label = command[2].split("/", 2)[-1]
            if label in active:
                return subprocess.CompletedProcess(
                    command, 0, "    state = running\n", "")
            return subprocess.CompletedProcess(command, 3, "", "absent")
        if command[1] == "submit":
            active.add(command[command.index("-l") + 1])
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1] == "bootout":
            active.discard(command[2].split("/", 2)[-1])
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    backend = life.LaunchdWatcherBackend(
        repo=ROOT, runpod_key_path=pod_state,
        run_command=run, sleep=lambda _seconds: None)
    handle = backend.start_watch(
        record_path=record, pod_state_path=pod_state,
        lifecycle_state_path=lifecycle)
    submit = next(command for command in commands if command[1] == "submit")
    assert "/usr/bin/caffeinate" in submit and "-dimsu" in submit
    assert f"SC_RUNPOD_KEY_PATH={pod_state.resolve()}" in submit
    assert "watch" in submit and str(pod_state.resolve()) in submit
    assert handle["mode"] == "ordinary"
    lifecycle.write_bytes(
        b'{"force_cleanup_reason":"test","status":"FORCE_CLEANUP"}\n')
    backend.request_cleanup(
        lifecycle_state_path=lifecycle, reason="test")
    emergency = backend.start_emergency(
        record_path=record, pod_state_path=pod_state,
        lifecycle_state_path=lifecycle)
    emergency_submit = [command for command in commands
                        if command[1] == "submit"][-1]
    assert "emergency-cleanup" in emergency_submit
    assert emergency["mode"] == "emergency"


def test_launchd_backend_fails_closed_when_submit_does_not_run(tmp_path):
    record, pod_state, lifecycle = _watchdog_fixture(tmp_path)

    def run(command, **_kwargs):
        if command[1] == "print":
            return subprocess.CompletedProcess(command, 3, "", "absent")
        return subprocess.CompletedProcess(command, 1, "", "failed")

    backend = life.LaunchdWatcherBackend(
        repo=ROOT, runpod_key_path=pod_state,
        run_command=run, sleep=lambda _seconds: None)
    with pytest.raises(life.V13LifecycleError, match="submission failed"):
        backend.start_watch(
            record_path=record, pod_state_path=pod_state,
            lifecycle_state_path=lifecycle)


def test_cli_probe_exits_and_harvest_manifest_are_machine_clean(tmp_path):
    lifecycle = tmp_path / "lifecycle.json"
    lifecycle.write_text(json.dumps({"status": "JOB_STARTED"}))
    remote = tmp_path / "remote.json"
    for state, expected in (("RUNNING", 0), ("COMPLETE", 20), ("DEAD", 21)):
        remote.write_text(json.dumps({"state": state}))
        completed = subprocess.run([
            sys.executable, str(CLI), "job-probe",
            "--lifecycle-state", str(lifecycle),
            "--remote-state", str(remote),
        ], cwd=ROOT, text=True, capture_output=True, check=False)
        assert completed.returncode == expected
        assert completed.stdout == "" and completed.stderr == ""

    root = tmp_path / "pulled"
    for category in life.HARVEST_ROOTS:
        path = root / category / "one.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    output = tmp_path / "harvest.json"
    completed = subprocess.run([
        sys.executable, str(CLI), "harvest-manifest",
        "--root", str(root), "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["file_count"] == 4
    assert output.is_file()


def test_live_watchdog_probe_turns_stale_running_state_into_dead(
        tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("stage_t_lifecycle_cli", CLI)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    lifecycle = tmp_path / "lifecycle.json"
    pod_state = tmp_path / "pod-state.json"
    key = tmp_path / "ssh-key"
    lifecycle.write_text('{"status":"JOB_STARTED"}\n')
    pod_state.write_text('{"id":"pod_stage_t_1"}\n')
    key.write_text("key\n")

    class Provider:
        pass

    monkeypatch.setattr(cli, "RunPodProvider", lambda **_kwargs: Provider())
    monkeypatch.setattr(
        cli, "_endpoint", lambda _provider, _state: ("203.0.113.8", 2222))

    real_run = subprocess.run

    def run(command, **_kwargs):
        assert "kill -0" in command[-1]
        assert "/workspace/powered-v13-stage-t/job.pid" in command[-1]
        syntax = real_run(
            ["bash", "-n", "-c", command[-1]], text=True,
            capture_output=True, check=False)
        assert syntax.returncode == 0, syntax.stderr
        return subprocess.CompletedProcess(
            command, 0, '{"state":"DEAD"}\n', "")

    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(SystemExit) as stopped:
        cli.command_watchdog_probe(argparse.Namespace(
            lifecycle_state=lifecycle, pod_state=pod_state, ssh_key=key))
    assert stopped.value.code == 21


def test_concrete_supervisor_invokes_emergency_after_unexpected_watcher_death(
        tmp_path):
    record = tmp_path / "watchdog.json"
    state = tmp_path / "lifecycle.json"
    pod_state = tmp_path / "pod-state.json"
    record.write_text("{}")
    state.write_text("{}")
    pod_state.write_text("{}")

    class Processes:
        def __init__(self):
            self.calls = []
            self.reads = [
                {"status": "TERMINATING", "termination_reason": "job_complete",
                 "harvest": {"succeeded": True}},
                {"status": "TERMINATED_HARVESTED",
                 "termination_reason": "job_complete",
                 "harvest": {"succeeded": True},
                 "per_pod_404_observed": True,
                 "active_inventory_absent_observed": True},
            ]

        def start_watch(self, **kwargs):
            self.calls.append("watch")
            return "ordinary"

        def start_emergency(self, **kwargs):
            self.calls.append("emergency")
            return "emergency"

        def request_cleanup(self, **kwargs):
            self.calls.append(("request", kwargs["reason"]))

        def wait(self, handle, *, timeout_seconds):
            self.calls.append(("wait", handle))
            return 1 if handle == "ordinary" else 0

        def read_record(self, _path):
            return self.reads.pop(0)

    processes = Processes()
    supervisor = life.RecoveringWatcherSupervisor(processes)
    handle = supervisor.start(
        record_path=record, pod_state_path=pod_state,
        lifecycle_state_path=state)
    result = supervisor.wait_terminal(handle, timeout_seconds=100)
    assert result["status"] == "TERMINATED_HARVESTED"
    assert processes.calls == [
        "watch", ("wait", "ordinary"), "emergency", ("wait", "emergency")]
