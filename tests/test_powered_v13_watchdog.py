from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import subprocess
import urllib.error

import pytest

import powered_v13_watchdog as watchdog


SHA = "a" * 64


def _record(**changes):
    values = {
        "pod_id": "pod_stage_t_1",
        "created_cost_per_hr_usd": "1.2",
        "prior_stage_t_spend_usd": "0",
        "provider_clock_started_epoch": 1_000_000,
        "pod_state_sha256": SHA,
        "create_response_sha256": "b" * 64,
        "job_sha256": "c" * 64,
        "release_receipt_sha256": "d" * 64,
        "job_probe_command": ["probe", "pod_stage_t_1"],
        "harvest_command": ["harvest", "pod_stage_t_1"],
    }
    values.update(changes)
    return watchdog.build_record(**values)


def _http_404() -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://provider.invalid/pods/id", 404, "gone", {}, None)


def _harvest(*_args):
    return {
        "attempted_epoch": 1_000_010,
        "returncode": 0,
        "stdout_sha256": "e" * 64,
        "stderr_sha256": "f" * 64,
        "timed_out": False,
        "succeeded": True,
    }


class FakeBackend:
    def __init__(self, *, get_after_delete="404", inventory_after_delete=()):
        self.deleted = False
        self.delete_calls = 0
        self.get_after_delete = get_after_delete
        self.inventory_after_delete = tuple(inventory_after_delete)

    def get_pod(self, pod_id):
        if self.deleted:
            if self.get_after_delete == "404":
                raise _http_404()
            if isinstance(self.get_after_delete, BaseException):
                raise self.get_after_delete
        return {"id": pod_id, "costPerHr": 1.2}

    def active_pod_ids(self):
        if self.deleted:
            return list(self.inventory_after_delete)
        return ["pod_stage_t_1"]

    def delete_pod(self, _pod_id):
        self.delete_calls += 1
        self.deleted = True


def test_record_uses_smaller_literal_time_or_rate_derived_spend_deadline():
    record = _record()
    assert record["bounded_provider_seconds"] == 3300
    assert record["hard_deadline_epoch"] == 1_003_300
    assert record["delete_trigger_epoch"] == 1_003_180
    assert (Decimal(record["created_cost_per_hr_usd"])
            * Decimal(record["bounded_provider_seconds"]) / Decimal(3600)
            <= Decimal("1.50"))

    expensive = _record(created_cost_per_hr_usd="2")
    assert expensive["bounded_provider_seconds"] == 2700
    assert expensive["hard_deadline_epoch"] == 1_002_700
    assert Decimal("2") * Decimal(2700) / Decimal(3600) == Decimal("1.50")

    carried = _record(
        created_cost_per_hr_usd="1.2", prior_stage_t_spend_usd="0.5")
    assert carried["bounded_provider_seconds"] == 3000


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"created_cost_per_hr_usd": "1.20"}, "canonical"),
        ({"prior_stage_t_spend_usd": "1.49",
          "created_cost_per_hr_usd": "2"}, "safe cleanup"),
        ({"provider_clock_started_epoch": True}, "provider clock"),
        ({"job_probe_command": []}, "job probe"),
    ],
)
def test_record_rejects_coercion_or_unsafe_remaining_window(changes, match):
    with pytest.raises(watchdog.V13WatchdogError, match=match):
        _record(**changes)


def test_record_rejects_literal_cap_or_deadline_tampering():
    record = _record()
    for field, value in (
        ("max_total_spend_usd", "2.00"),
        ("max_provider_seconds", 3301),
        ("delete_lead_seconds", 1),
        ("hard_deadline_epoch", record["hard_deadline_epoch"] + 1),
    ):
        altered = deepcopy(record)
        altered[field] = value
        with pytest.raises(watchdog.V13WatchdogError):
            watchdog.validate_record(altered)


@pytest.mark.parametrize(
    "job_state,pod,error,now_delta,expected",
    [
        ("RUNNING", {"id": "pod_stage_t_1", "costPerHr": 1.2}, None, 0,
         "CONTINUE"),
        ("COMPLETE", {"id": "pod_stage_t_1", "costPerHr": 1.2}, None, 0,
         "STOP_JOB_COMPLETE"),
        ("DEAD", {"id": "pod_stage_t_1", "costPerHr": 1.2}, None, 0,
         "STOP_PROCESS_DEATH"),
        ("AMBIGUOUS", {"id": "pod_stage_t_1", "costPerHr": 1.2}, None, 0,
         "HOLD_JOB_AMBIGUOUS"),
        ("RUNNING", {"id": "pod_stage_t_1", "costPerHr": 1.21}, None, 0,
         "STOP_RATE_INCREASE"),
        ("RUNNING", None, RuntimeError("network"), 0,
         "HOLD_PROVIDER_AMBIGUOUS"),
        ("RUNNING", None, _http_404(), 0,
         "STOP_PROVIDER_GONE"),
        ("RUNNING", None, RuntimeError("network"), 3180,
         "STOP_DEADLINE"),
    ],
)
def test_guard_decision_fault_matrix(job_state, pod, error, now_delta, expected):
    record = _record()
    observed = watchdog.guard_decision(
        record=record,
        now_epoch=record["provider_clock_started_epoch"] + now_delta,
        job_state=job_state,
        provider_pod=pod,
        provider_error=error,
    )
    assert observed == expected


def test_probe_job_has_exact_running_complete_dead_and_ambiguous_exits():
    def fake(returncode):
        def run(*_args, **_kwargs):
            return subprocess.CompletedProcess([], returncode, b"", b"")
        return run

    assert watchdog.probe_job(["probe"], run=fake(0)) == "RUNNING"
    assert watchdog.probe_job(["probe"], run=fake(20)) == "COMPLETE"
    assert watchdog.probe_job(["probe"], run=fake(21)) == "DEAD"
    assert watchdog.probe_job(["probe"], run=fake(9)) == "AMBIGUOUS"


def test_happy_path_harvests_then_requires_404_and_absent_inventory(tmp_path):
    path = tmp_path / "guard.json"
    watchdog.write_record_exclusive(path, _record())
    backend = FakeBackend()
    result = watchdog.watch(
        record_path=path,
        backend=backend,
        clock=lambda: 1_000_010,
        sleep=lambda _seconds: None,
        probe=lambda _command: "COMPLETE",
        harvest=_harvest,
        max_cycles=1,
    )
    assert result["status"] == "TERMINATED_HARVESTED"
    assert result["termination_reason"] == "job_complete"
    assert result["per_pod_404_observed"] is True
    assert result["active_inventory_absent_observed"] is True
    assert backend.delete_calls == 1
    assert watchdog.read_record(path) == result


def test_process_death_harvest_failure_still_deletes_and_is_terminal(tmp_path):
    path = tmp_path / "guard.json"
    watchdog.write_record_exclusive(path, _record())
    backend = FakeBackend()

    def failed_harvest(_command, epoch):
        row = _harvest()
        row.update({"attempted_epoch": epoch, "returncode": 23,
                    "succeeded": False})
        return row

    result = watchdog.watch(
        record_path=path,
        backend=backend,
        clock=lambda: 1_000_010,
        sleep=lambda _seconds: None,
        probe=lambda _command: "DEAD",
        harvest=failed_harvest,
        max_cycles=1,
    )
    assert result["status"] == "TERMINATED_HARVEST_FAILED"
    assert result["termination_reason"] == "process_death"
    assert backend.delete_calls == 1


@pytest.mark.parametrize(
    "get_after,inventory,confirmed",
    [
        ("404", ("pod_stage_t_1",), False),
        ("present", (), False),
        ("404", (), True),
    ],
)
def test_cleanup_requires_same_cycle_dual_provider_confirmation(
        tmp_path, get_after, inventory, confirmed):
    record = _record()
    record.update({
        "status": "TERMINATING",
        "termination_reason": "fault_injection",
        "harvest": _harvest(),
    })
    record = watchdog.validate_record(record)
    backend = FakeBackend(
        get_after_delete=get_after, inventory_after_delete=inventory)
    result, observed = watchdog.cleanup_once(
        backend=backend, record=record, now_epoch=1_000_011)
    assert observed is confirmed
    if confirmed:
        assert result["status"] == "TERMINATED_HARVESTED"
    else:
        assert result["status"] == "TERMINATING"


def test_deadline_overrides_provider_and_job_ambiguity_and_cleans_up(tmp_path):
    path = tmp_path / "guard.json"
    record = _record()
    watchdog.write_record_exclusive(path, record)
    backend = FakeBackend()
    result = watchdog.watch(
        record_path=path,
        backend=backend,
        clock=lambda: record["delete_trigger_epoch"],
        sleep=lambda _seconds: None,
        probe=lambda _command: "AMBIGUOUS",
        harvest=_harvest,
        max_cycles=1,
    )
    assert result["status"] == "TERMINATED_HARVESTED"
    assert result["termination_reason"] == "deadline"


def test_watchdog_refuses_to_overwrite_existing_record(tmp_path):
    path = tmp_path / "guard.json"
    watchdog.write_record_exclusive(path, _record())
    with pytest.raises(FileExistsError):
        watchdog.write_record_exclusive(path, _record())


def test_independent_supervisor_can_resume_after_watcher_exit(tmp_path):
    path = tmp_path / "guard.json"
    watchdog.write_record_exclusive(path, _record())
    backend = FakeBackend()
    first = watchdog.watch(
        record_path=path,
        backend=backend,
        clock=lambda: 1_000_010,
        sleep=lambda _seconds: None,
        probe=lambda _command: "RUNNING",
        harvest=_harvest,
        max_cycles=1,
    )
    assert first["status"] == "WATCHING"
    assert backend.delete_calls == 0

    recovered = watchdog.watch(
        record_path=path,
        backend=backend,
        clock=lambda: 1_000_011,
        sleep=lambda _seconds: None,
        probe=lambda _command: "DEAD",
        harvest=_harvest,
        max_cycles=1,
    )
    assert recovered["status"] == "TERMINATED_HARVESTED"
    assert recovered["termination_reason"] == "process_death"
    assert backend.delete_calls == 1
