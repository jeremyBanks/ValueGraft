from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import subprocess
import sys
import urllib.error

import pytest

import powered_v13_lifecycle as life


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_powered_v13_stage_t_lifecycle.py"


class Clock:
    def __init__(self):
        self.value = 1_000_000

    def __call__(self):
        self.value += 1
        return self.value


class FakeProvider:
    def __init__(self, clock, actions=("success",), log=None):
        self.clock = clock
        self.actions = list(actions)
        self.log = log if log is not None else []
        self.creates = 0
        self.active = []
        self.deleted = set()

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
        response = {
            "id": pod_id, "costPerHr": "1.2", "cloudType": "SECURE",
            "gpuCount": 1,
        }
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
            "cuda_available": True, "gpu_uuid": "GPU-fixed",
            "driver_version": "580.65",
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
                 reason="job_complete", dual=True, death=None, log=None):
        self.provider = provider
        self.terminal_status = terminal_status
        self.reason = reason
        self.dual = dual
        self.death = death
        self.log = log if log is not None else []
        self.harvests = 0
        self.cleanup_requests = 0

    def start(self, *, record_path, state_path):
        assert record_path.is_file()
        assert state_path.is_file()
        self.log.append(("watcher_start", record_path.name))
        return {"record_path": record_path}

    def request_cleanup(self, handle, *, reason):
        self.cleanup_requests += 1
        self.log.append(("cleanup_request", reason))

    def wait_terminal(self, handle, *, timeout_seconds):
        self.log.append(("wait", timeout_seconds))
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
         dual=True, death=None):
    clock = Clock()
    log = []
    provider = FakeProvider(clock, provider_actions, log)
    transport = FakeTransport(
        admissions, sync_error=sync_error, launch_error=launch_error, log=log)
    supervisor = FakeSupervisor(
        provider, terminal_status=terminal_status, reason=reason,
        dual=dual, death=death, log=log)
    repo, release = _release(tmp_path)
    runner = life.StageTLifecycle(
        repo=repo, session_root=tmp_path / "session", release=release,
        provider=provider, transport=transport, supervisor=supervisor,
        clock=clock, prior_stage_t_spend_usd="0",
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
    create = next(row for row in log if isinstance(row, tuple)
                  and row[0] == "create")
    attempt = result["attempts"][0]
    assert create[2] == attempt["clock_started_epoch"]
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
    for relative in (life.JOB_PATH, "contract.json", "PREREG.md"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n")
    commit = "a" * 40
    manifest_path = "release/stage-t.json"
    inventory = [{"path": path} for path in sorted((
        "PREREG.md", "contract.json", life.JOB_PATH))]
    manifest = {"inventory": inventory}
    (repo / manifest_path).write_bytes(life.canonical_json_bytes(manifest) + b"\n")
    receipt = tmp_path / "receipts" / "powered-v13-stage-t-launch-receipt.json"
    receipt.parent.mkdir()
    receipt_doc = {"design_id": life.DESIGN_ID, "stage": "TECHNICAL_CANARY",
                   "authorization_commit": commit, "manifest_path": manifest_path}
    receipt.write_bytes(life.canonical_json_bytes(receipt_doc) + b"\n")
    report = tmp_path / "import.json"
    report_doc = {
        "schema": life.IMPORT_REPORT_SCHEMA, "design_id": life.DESIGN_ID,
        "stage": "TECHNICAL_CANARY", "status": "PASS", "semantic_n": 0,
        "production_entropy_requested": False, "execution_roots": [life.JOB_PATH],
        "contract_path": "contract.json",
    }
    report_doc["report_sha256"] = life.sha256_bytes(
        life.canonical_json_bytes(report_doc))
    report.write_bytes(life.canonical_json_bytes(report_doc) + b"\n")
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
    verified = life.verify_release_binding(
        repo=args[0], expected_authorization_commit=args[1],
        manifest_path=args[2], receipt_path=args[3],
        expected_receipt_sha256=args[4], import_report_path=args[5],
        expected_import_report_sha256=args[6], verify_checkout=args[7])
    assert verified.sync_paths == tuple(sorted((
        "PREREG.md", "contract.json", "release/stage-t.json", life.JOB_PATH)))


def test_release_binding_rejects_production_path(tmp_path):
    args = list(_binding_fixture(tmp_path))
    manifest_path = args[0] / args[2]
    manifest = json.loads(manifest_path.read_text())
    forbidden = sorted(life.FORBIDDEN_INVENTORY_PATHS)[0]
    manifest["inventory"].append({"path": forbidden})
    manifest["inventory"] = sorted(manifest["inventory"], key=lambda row: row["path"])
    manifest_path.write_bytes(life.canonical_json_bytes(manifest) + b"\n")

    def verify_checkout(**_kwargs):
        evidence = args[7](**_kwargs)
        evidence["authorization"]["manifest_sha256"] = life.file_sha256(manifest_path)
        return evidence
    with pytest.raises(life.V13LifecycleError, match="production-only"):
        life.verify_release_binding(
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
        value = {"id": "pod_stage_t_1", "costPerHr": "1.2",
                 "cloudType": "SECURE", "gpuCount": 1}
        self.state_path.write_text(json.dumps(value))
        return value


def test_runpod_adapter_reuses_pod_api_and_classifies_capacity(tmp_path):
    state = tmp_path / "pod.json"
    module = FakePodModule(state)
    provider = life.RunPodProvider(state_path=state, module=module)
    assert provider.safe_snapshot() == {
        "balance_usd": "57.12", "spend_limit_usd": "80",
        "active_pod_ids": [],
    }
    allocation = provider.create_secure_a100()
    assert allocation.pod_id == "pod_stage_t_1"
    with pytest.raises(life.PodNotFound):
        provider.get_pod("pod_stage_t_1")

    no_capacity = FakePodModule(
        state, create_error=urllib.error.HTTPError(
            "x", 500, "none", {}, None))
    provider = life.RunPodProvider(state_path=state, module=no_capacity)
    with pytest.raises(life.NoCapacity):
        provider.create_secure_a100()


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


def test_concrete_supervisor_invokes_emergency_after_unexpected_watcher_death(
        tmp_path):
    record = tmp_path / "watchdog.json"
    state = tmp_path / "lifecycle.json"
    record.write_text("{}")
    state.write_text("{}")

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
    handle = supervisor.start(record_path=record, state_path=state)
    result = supervisor.wait_terminal(handle, timeout_seconds=100)
    assert result["status"] == "TERMINATED_HARVESTED"
    assert processes.calls == [
        "watch", ("wait", "ordinary"), "emergency", ("wait", "emergency")]
