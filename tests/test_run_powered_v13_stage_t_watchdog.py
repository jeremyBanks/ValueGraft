from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import powered_v13_watchdog as watchdog


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_powered_v13_stage_t_watchdog.py"


def _files(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    state = tmp_path / "state.json"
    response = tmp_path / "response.json"
    job = tmp_path / "job.sh"
    receipt = tmp_path / "receipt.json"
    state.write_text(json.dumps({"id": "pod_stage_t_1", "costPerHr": 1.2}))
    response.write_text(json.dumps({"id": "pod_stage_t_1", "costPerHr": 1.2}))
    job.write_text("#!/bin/sh\nexit 0\n")
    receipt.write_text('{"receipt":"fixed"}\n')
    return state, response, job, receipt


def _command(tmp_path: Path, *extra: str):
    state, response, job, receipt = _files(tmp_path)
    record = tmp_path / "guard.json"
    return [
        sys.executable,
        str(SCRIPT),
        "init",
        "--state", str(state),
        "--create-response", str(response),
        "--job", str(job),
        "--release-receipt", str(receipt),
        "--record", str(record),
        "--prior-stage-t-spend-usd", "0",
        "--provider-clock-started-epoch", "1000000",
        "--job-probe-command-json", '["probe","pod_stage_t_1"]',
        "--harvest-command-json", '["harvest","pod_stage_t_1"]',
        *extra,
    ], record


def test_init_binds_all_external_files_and_exclusive_record(tmp_path):
    command, record = _command(tmp_path)
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["pod_id"] == "pod_stage_t_1"
    observed = watchdog.read_record(record)
    assert observed["status"] == "ALLOCATED"
    assert observed["bounded_provider_seconds"] == 3300
    assert observed["job_probe_command"] == ["probe", "pod_stage_t_1"]
    assert observed["harvest_command"] == ["harvest", "pod_stage_t_1"]

    repeated = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert repeated.returncode == 2
    assert "STAGE-T WATCHDOG ERROR" in repeated.stderr
    assert "Traceback" not in repeated.stderr
    assert record.is_file()


def test_init_rejects_state_response_identity_or_rate_disagreement(tmp_path):
    command, record = _command(tmp_path)
    response = Path(command[command.index("--create-response") + 1])
    response.write_text(json.dumps({"id": "other_pod", "costPerHr": 1.2}))
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "IDs differ" in completed.stderr
    assert not record.exists()

    command, record = _command(tmp_path / "rate")
    response = Path(command[command.index("--create-response") + 1])
    response.write_text(json.dumps({"id": "pod_stage_t_1", "costPerHr": 1.3}))
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "rates differ" in completed.stderr
    assert not record.exists()


def test_init_rejects_symlinked_bound_inputs(tmp_path):
    command, record = _command(tmp_path)
    job = Path(command[command.index("--job") + 1])
    target = tmp_path / "real-job.sh"
    job.rename(target)
    job.symlink_to(target)
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "symlinked" in completed.stderr
    assert not record.exists()
