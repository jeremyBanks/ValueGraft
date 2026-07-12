from __future__ import annotations

import os
import pytest
from pathlib import Path
import subprocess
import textwrap

import pod
from pod_admission import PodAdmissionError, parse_gpu_row, validate_gpu


def test_create_body_applies_provider_cuda_filter(monkeypatch):
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13.0")
    monkeypatch.setenv("SC_POD_CLOUD", "SECURE")
    body = pod.create_body("NVIDIA A100 80GB PCIe")
    assert body["allowedCudaVersions"] == ["13.0"]
    assert body["cloudType"] == "SECURE"
    assert body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe"]
    monkeypatch.setenv("SC_POD_NAME", "v12tech3-1")
    assert pod.create_body("gpu")["name"] == "v12tech3-1"
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13.0,13.0,12.9")
    assert pod.create_body("gpu")["allowedCudaVersions"] == ["13.0", "12.9"]


def test_create_body_rejects_invalid_cuda_filter(monkeypatch):
    monkeypatch.setenv("SC_POD_ALLOWED_CUDA", "13")
    with pytest.raises(ValueError, match="invalid SC_POD_ALLOWED_CUDA"):
        pod.create_body("gpu")


def test_create_requires_provider_pod_id_before_writing_state(
        monkeypatch, tmp_path):
    state = tmp_path / "pod.json"
    monkeypatch.setattr(pod, "STATE", state)
    monkeypatch.setattr(pod, "ensure_ssh_key", lambda: "key")
    monkeypatch.setattr(pod, "api", lambda *args, **kwargs: {"error": "none"})
    with pytest.raises(RuntimeError, match="did not contain a pod ID"):
        pod.create("gpu")
    assert not state.exists()


def test_gpu_admission_requires_one_exact_compatible_gpu():
    row = "NVIDIA A100 80GB PCIe, 580.159.03, 81920"
    assert parse_gpu_row(row) == {
        "gpu_name": "NVIDIA A100 80GB PCIe",
        "driver_version": "580.159.03",
        "memory_total_mib": 81920,
    }
    assert validate_gpu(
        row, expected_name="NVIDIA A100 80GB PCIe",
        minimum_driver="580.65.06", minimum_memory_mib=80000)


@pytest.mark.parametrize("row, message", [
    ("NVIDIA A100 80GB PCIe, 550.90.12, 81920", "driver"),
    ("NVIDIA A100-SXM4-80GB, 580.159.03, 81920", "GPU name"),
    ("NVIDIA A100 80GB PCIe, 580.159.03, 40960", "GPU memory"),
    ("NVIDIA A100 80GB PCIe, 580.159.03, 81920\nextra, 580.1, 1",
     "exactly one"),
])
def test_gpu_admission_rejects_incompatible_hosts(row, message):
    with pytest.raises(PodAdmissionError, match=message):
        validate_gpu(
            row, expected_name="NVIDIA A100 80GB PCIe",
            minimum_driver="580.65.06", minimum_memory_mib=80000)


def test_exact_launch_wrapper_binds_cuda_driver_and_commit_mechanically():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts/launch_coherent_canary_v12_technical.sh").read_text()
    launcher = (root / "scripts/launch_pod.sh").read_text()
    job = (root / "scripts/job_coherent_canary_v12_technical.sh").read_text()
    assert 'HEAD_COMMIT="$(git rev-parse HEAD)"' in wrapper
    assert "SC_POD_ALLOWED_CUDA=13.0" in wrapper
    assert "SC_MIN_NVIDIA_DRIVER=580.65.06" in wrapper
    assert 'SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"' in wrapper
    assert "SC_TERMINATE_ON_LAUNCH_FAILURE=1" in wrapper
    assert "SC_REQUIRE_HF_TOKEN_DEPLOY=1" in wrapper
    assert "SC_SSH_WAIT_ATTEMPTS=20" in wrapper
    assert "MAX_ADMISSION_ATTEMPTS" in wrapper
    assert '"$status" -eq 85' in wrapper
    assert '"$status" -eq 86' in wrapper
    assert "status=$status); not retrying" in wrapper
    assert launcher.index("nvidia-smi --query-gpu=name,driver_version,memory.total") < \
        launcher.index("apt-get update")
    assert "src/pod_admission.py" in launcher
    assert "trap cleanup_on_failure EXIT" in launcher
    assert "refusing any retry" in launcher
    assert "AUTO_CLEANUP_ARMED=0" in launcher
    assert "bootstrap failed after host admission" in launcher
    assert "required remote Hugging Face token is absent/empty" in launcher
    assert 'exit 86' in launcher
    assert 'git checkout --quiet -B trunk "$EXPECTED_COMMIT"' in job
    assert 'git merge-base --is-ancestor "$EXPECTED_COMMIT" origin/trunk' in job


def test_termination_success_is_not_reclassified_by_balance_outage(
        monkeypatch, tmp_path, capsys):
    state = tmp_path / "pod.json"
    state.write_text('{"id": "pod-1"}')
    calls = []

    def fake_api(method, path, body=None):
        calls.append((method, path, body))
        return {}

    monkeypatch.setattr(pod, "STATE", state)
    monkeypatch.setattr(pod, "api", fake_api)
    monkeypatch.setattr(
        pod, "balance", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    pod.terminate()
    assert calls == [("DELETE", "/pods/pod-1", None)]
    assert "deletion succeeded but balance lookup failed" in capsys.readouterr().err


def _executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip())
    path.chmod(0o755)


def _launcher_fixture(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    root = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    scratch = tmp_path / "scratch"
    (root / "src").mkdir(parents=True)
    fake_bin.mkdir()
    scratch.mkdir()
    (root / "src" / "placeholder.py").write_text("x = 1\n")
    (root / ".huggingface_key").write_text("test-token\n")
    job = root / "job.sh"
    _executable(job, "#!/bin/sh\nexit 0\n")
    event_log = tmp_path / "events.log"

    _executable(fake_bin / "uv", r'''#!/bin/sh
case "$*" in
  *"src/pod.py create"*)
    echo create >> "$FAKE_EVENT_LOG"
    rc="${FAKE_CREATE_RC:-0}"
    if [ "$rc" -eq 0 ]; then printf '{"id":"pod-test"}\n' > "$SC_POD_STATE"; fi
    exit "$rc" ;;
  *"src/pod.py status"*)
    echo status >> "$FAKE_EVENT_LOG"
    count_file="${FAKE_STATUS_COUNT_FILE}"
    count=0; [ ! -f "$count_file" ] || count=$(sed -n '1p' "$count_file")
    count=$((count + 1)); printf '%s\n' "$count" > "$count_file"
    if [ "${FAKE_STATUS_ALWAYS_FAIL:-0}" -eq 1 ]; then exit 1; fi
    if [ "${FAKE_STATUS_FAIL_ONCE:-0}" -eq 1 ] && [ "$count" -eq 1 ]; then exit 1; fi
    if [ "${FAKE_NO_ENDPOINT:-0}" -eq 1 ]; then
      printf '{"publicIp":null,"portMappings":{}}\n'; exit 0
    fi
    printf '{"publicIp":"127.0.0.1","portMappings":{"22":"2222"}}\n'
    exit 0 ;;
  *"src/pod.py terminate"*)
    echo terminate >> "$FAKE_EVENT_LOG"
    exit "${FAKE_TERMINATE_RC:-0}" ;;
  *"src/pod_admission.py"*) echo admission >> "$FAKE_EVENT_LOG"; exit 0 ;;
esac
exit 1
''')
    _executable(fake_bin / "ssh", r'''#!/bin/sh
case "$*" in
  *"--query-gpu=name,driver_version,memory.total"*)
    printf 'NVIDIA A100 80GB PCIe, 580.159.03, 81920\n'; exit 0 ;;
  *"apt-get update"*)
    [ "${FAKE_BOOTSTRAP_FAIL:-0}" -eq 0 ]; exit $? ;;
  *"nohup bash job.sh"*) echo remote-launch >> "$FAKE_EVENT_LOG"; echo job-launched; exit 0 ;;
  *"ALIVE="*) echo 'ALIVE=1 LINES=1 CRASH=0 DONE=0'; exit 0 ;;
esac
exit 0
''')
    _executable(fake_bin / "rsync", r'''#!/bin/sh
case "$*" in
  *".huggingface_key"*)
    [ "${FAKE_TOKEN_RSYNC_FAIL:-0}" -eq 0 ]; exit $? ;;
esac
exit 0
''')
    _executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")

    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SC_REPO_ROOT": str(root),
        "SC_POD_SCRATCH": str(scratch),
        "SC_SKIP_PREFLIGHT": "test fixture",
        "SC_TERMINATE_ON_LAUNCH_FAILURE": "1",
        "SC_REQUIRE_HF_TOKEN_DEPLOY": "1",
        "SC_SSH_WAIT_ATTEMPTS": "2",
        "SC_EXPECTED_GPU_NAME": "NVIDIA A100 80GB PCIe",
        "SC_MIN_NVIDIA_DRIVER": "580.65.06",
        "SC_MIN_GPU_MEMORY_MIB": "80000",
        "FAKE_EVENT_LOG": str(event_log),
        "FAKE_STATUS_COUNT_FILE": str(tmp_path / "status.count"),
    })
    return env, job, event_log


def _run_launcher(env: dict[str, str], job: Path) -> subprocess.CompletedProcess[str]:
    launcher = Path(__file__).resolve().parents[1] / "scripts/launch_pod.sh"
    return subprocess.run(
        ["bash", str(launcher), "fixture-pod", str(job),
         "NVIDIA A100 80GB PCIe"],
        env=env, capture_output=True, text=True, check=False)


def test_launcher_retries_status_read_without_losing_cleanup(tmp_path):
    env, job, events = _launcher_fixture(tmp_path)
    env["FAKE_STATUS_FAIL_ONCE"] = "1"
    result = _run_launcher(env, job)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = events.read_text().splitlines()
    assert observed.count("status") == 2
    assert observed.count("remote-launch") == 1
    assert "terminate" not in observed


def test_launcher_stops_after_unreadable_provider_status(tmp_path):
    env, job, events = _launcher_fixture(tmp_path)
    env["FAKE_STATUS_ALWAYS_FAIL"] = "1"
    result = _run_launcher(env, job)
    assert result.returncode == 1, result.stdout + result.stderr
    observed = events.read_text().splitlines()
    assert observed.count("status") == 2
    assert observed.count("terminate") == 1
    assert "remote-launch" not in observed


def test_launcher_classifies_attested_missing_endpoint_as_admission(tmp_path):
    env, job, events = _launcher_fixture(tmp_path)
    env["FAKE_NO_ENDPOINT"] = "1"
    result = _run_launcher(env, job)
    assert result.returncode == 86, result.stdout + result.stderr
    observed = events.read_text().splitlines()
    assert observed.count("status") == 2
    assert observed.count("terminate") == 1


@pytest.mark.parametrize("failure, terminate_rc, expected_rc", [
    ("bootstrap", "0", 1),
    ("bootstrap", "9", 87),
    ("credential", "0", 1),
])
def test_launcher_cleans_one_known_pod_and_never_retries_nonadmission_failure(
        tmp_path, failure, terminate_rc, expected_rc):
    env, job, events = _launcher_fixture(tmp_path)
    env["FAKE_TERMINATE_RC"] = terminate_rc
    if failure == "bootstrap":
        env["FAKE_BOOTSTRAP_FAIL"] = "1"
    else:
        env["FAKE_TOKEN_RSYNC_FAIL"] = "1"
    result = _run_launcher(env, job)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    observed = events.read_text().splitlines()
    assert observed.count("create") == 1
    assert observed.count("terminate") == 1
    assert "remote-launch" not in observed


@pytest.mark.parametrize("create_rc", [1, 85])
def test_launcher_does_not_cleanup_or_retry_allocation_without_known_pod(
        tmp_path, create_rc):
    env, job, events = _launcher_fixture(tmp_path)
    env["FAKE_CREATE_RC"] = str(create_rc)
    result = _run_launcher(env, job)
    assert result.returncode == create_rc
    assert events.read_text().splitlines() == ["create"]


def _wrapper_fixture(tmp_path: Path) -> tuple[dict[str, str], Path]:
    root = tmp_path / "wrapper-repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    event_log = tmp_path / "wrapper-events.log"
    count_file = tmp_path / "wrapper-count"
    _executable(scripts / "launch_pod.sh", r'''#!/bin/sh
count=0; [ ! -f "$FAKE_WRAPPER_COUNT" ] || count=$(sed -n '1p' "$FAKE_WRAPPER_COUNT")
count=$((count + 1)); printf '%s\n' "$count" > "$FAKE_WRAPPER_COUNT"
echo "$1" >> "$FAKE_WRAPPER_EVENTS"
rc=$(printf '%s' "$FAKE_WRAPPER_RCS" | cut -d, -f"$count")
exit "${rc:-1}"
''')
    (scripts / "job_coherent_canary_v12_technical.sh").write_text("#!/bin/sh\n")
    subprocess.run(["git", "init", "-q", "-b", "trunk"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "scripts"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()
    subprocess.run(["git", "update-ref", "refs/remotes/origin/trunk", head],
                   cwd=root, check=True)
    env = os.environ.copy()
    env.update({
        "SC_REPO_ROOT": str(root),
        "SC_ADMISSION_MAX_ATTEMPTS": "3",
        "FAKE_WRAPPER_COUNT": str(count_file),
        "FAKE_WRAPPER_EVENTS": str(event_log),
    })
    return env, event_log


@pytest.mark.parametrize("sequence, expected_rc, expected_calls", [
    ("85,0", 0, 2),
    ("86,0", 0, 2),
    ("1,0", 1, 1),
    ("87,0", 87, 1),
    ("86,86,86", 86, 3),
])
def test_exact_wrapper_retries_only_allocation_and_admission(
        tmp_path, sequence, expected_rc, expected_calls):
    env, events = _wrapper_fixture(tmp_path)
    env["FAKE_WRAPPER_RCS"] = sequence
    wrapper = Path(__file__).resolve().parents[1] / \
        "scripts/launch_coherent_canary_v12_technical.sh"
    result = subprocess.run(
        ["bash", str(wrapper), "fixture"], env=env,
        capture_output=True, text=True, check=False)
    assert result.returncode == expected_rc, result.stdout + result.stderr
    assert len(events.read_text().splitlines()) == expected_calls
