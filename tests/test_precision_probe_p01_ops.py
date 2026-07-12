from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import textwrap
import time


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "scripts/job_precision_probe_p01.sh"
LAUNCH = ROOT / "scripts/launch_precision_probe_p01.sh"
PULL = ROOT / "scripts/pull_precision_probe_p01.sh"


def text(path: Path) -> str:
    return path.read_text()


def test_p01_shell_entrypoints_parse_without_a_pod() -> None:
    result = subprocess.run(
        ["bash", "-n", str(JOB), str(LAUNCH), str(PULL)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    shellcheck = subprocess.run(
        ["sh", "-c", "command -v shellcheck"],
        capture_output=True,
        text=True,
        check=False,
    )
    if shellcheck.returncode == 0:
        checked = subprocess.run(
            ["shellcheck", "-x", str(JOB), str(LAUNCH), str(PULL)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def test_job_binds_exact_subject_checkout_and_isolated_stack() -> None:
    source = text(JOB)
    assert 'EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?' in source
    assert 'MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"' in source
    assert 'REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"' in source
    assert 'git checkout --quiet -B trunk "$EXPECTED_COMMIT"' in source
    assert 'git merge-base --is-ancestor "$EXPECTED_COMMIT" origin/trunk' in source
    assert 'uv python install 3.12.11' in source
    for requirement in (
        "torch==2.12.1",
        "transformers==4.57.6",
        "accelerate==1.14.0",
        "bitsandbytes==0.49.2",
        "huggingface-hub==0.36.2",
        "safetensors==0.8.0",
        "tokenizers==0.22.2",
    ):
        assert requirement in source
    assert "transformers==5.0.0" not in source


def test_job_gates_host_budget_and_invokes_only_the_p01_runner() -> None:
    source = text(JOB)
    assert 'torch.cuda.get_device_name(0) != "NVIDIA A100 80GB PCIe"' in source
    assert 'version(driver) < version("580.65.06")' in source
    assert 'int(memory) < 80000' in source
    assert 'torch.version.cuda != "13.0"' in source
    assert 'cap * float(rate) / 3600 > 4.0' in source
    assert 'not 1 <= cap <= 7200' in source
    assert 'timeout --signal=INT --kill-after=60s' in source
    assert 'scripts/run_precision_probe_p01.py' in source
    assert '--hourly-cost-usd "$PROVIDER_RATE"' in source
    assert '--provider-elapsed-seconds-at-start "$ELAPSED"' in source
    assert 'OPERATIONAL_INTERRUPT_ELAPSED=$((PROVIDER_CAP - RESERVE_SECONDS))' in source
    assert '--provider-wall-cap-seconds "$PROVIDER_CAP"' in source
    assert "run_coherent_canary_v12_treatment.py" not in source
    assert "verify_frozen_repository" not in source


def test_job_verifies_lossless_packages_and_receipts_partial_trees() -> None:
    source = text(JOB)
    assert "artifact_packager.py" in source
    assert '"$PY" "$PACKAGER" verify "$package"' in source
    assert "precision-probe-p01-outcome-*-raw_Qwen3-30B-A3B-Instruct-2507_" in source
    assert 'find "$CURRENT_RUN_DIR" -type f' in source
    assert "-path '*/.scratch/*.json'" in source
    assert '"recovery_raw_count": recovery_raw_count' in source
    assert '"outcome_package_count": outcome_package_count' in source
    assert '"formal_v12_decision_eligible": False' in source
    assert '"v12_reentry_authorized": False' in source
    assert '"component_reuse_does_not_inherit_v12_eligibility": True' in source
    assert 'completion["matched_runtime_gate"].get("status") == "PASS"' in source
    assert 'printf \'%s\\n\' "$RECEIPT" > "${POINTER}.tmp"' in source


def test_launcher_is_clean_pushed_secure_and_owns_provider_clock() -> None:
    source = text(LAUNCH)
    assert '[ "$(git branch --show-current)" = "trunk" ]' in source
    assert '[ -z "$(git status --porcelain)" ]' in source
    assert '[ "$HEAD_COMMIT" = "$(git rev-parse origin/trunk)" ]' in source
    assert "SC_POD_CLOUD=SECURE" in source
    assert "SC_POD_ALLOWED_CUDA=13.0" in source
    assert 'SC_EXPECTED_GPU_NAME="NVIDIA A100 80GB PCIe"' in source
    assert "SC_MIN_NVIDIA_DRIVER=580.65.06" in source
    assert "SC_MIN_GPU_MEMORY_MIB=80000" in source
    assert "SC_REQUIRE_HF_TOKEN_DEPLOY=1" in source
    assert "MAX_PROVIDER_SECONDS=7200" in source
    assert "MAX_PROVIDER_USD=4.00" in source
    assert "provider_clock_started_epoch" in source
    assert "launchctl submit" in source
    assert "launchctl print" in source
    assert "launchctl bootout" in source
    assert "state = running" in source
    assert 'trap \'remove_own_launchd_label "$SELF_WATCHDOG_LABEL" "$?"\' EXIT' in source
    assert 'if [ "$WATCHDOG_TERMINAL" -eq 1 ]; then' in source
    assert '"$CAFFEINATE" -dimsu' in source
    assert "nohup caffeinate" not in source
    assert "WATCHDOG_PID" not in source
    assert "cleanup_unwatched_allocation" in source
    assert 'WATCHDOG_CONFIRMED=1' in source
    assert "SC_ADMISSION_MAX_ATTEMPTS must be 1 or 2" in source


def test_watchdog_removes_own_launchd_label_after_normal_exit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    launchctl_log = tmp_path / "launchctl.log"
    pull_log = tmp_path / "pull.log"
    (root / "scripts").mkdir(parents=True)
    (root / "src").mkdir()
    fake_bin.mkdir()
    state = root / ".pod_fixture_state.json"
    state.write_text('{"id":"fixture"}\n')

    _executable(
        root / "scripts/pull_precision_probe_p01.sh",
        """
        #!/bin/sh
        printf '%s\n' "$*" >> "$FAKE_PULL_LOG"
        """,
    )
    _executable(
        fake_bin / "uv",
        """
        #!/bin/sh
        case "$*" in
          *"pod.py status"*)
            printf '%s\n' '{"publicIp":"127.0.0.1","portMappings":{"22":"2222"},"costPerHr":1.0}'
            ;;
          *"pod.py terminate"*)
            printf '%s\n' 'terminated fixture'
            ;;
          *) exit 9 ;;
        esac
        """,
    )
    _executable(
        fake_bin / "ssh",
        """
        #!/bin/sh
        printf '%s\n' 'results/precision_probe_p01/precision-probe-p01-receipt_Qwen3-30B-A3B-Instruct-2507_20260712T080000Z.json'
        """,
    )
    _executable(
        fake_bin / "launchctl",
        """
        #!/bin/sh
        printf '%s\n' "$*" >> "$FAKE_LAUNCHCTL_LOG"
        """,
    )

    label = "com.semantic-continuity.precision-p01.fixture"
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SC_REPO_ROOT": str(root),
        "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
        "FAKE_PULL_LOG": str(pull_log),
    })
    result = subprocess.run(
        [
            "bash",
            str(LAUNCH),
            "--watch",
            "fixture",
            str(state),
            str(int(time.time())),
            "3600",
            "1.0",
            label,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert pull_log.read_text().strip() == "fixture"
    assert (
        launchctl_log.read_text().strip()
        == f"bootout gui/{os.getuid()}/{label}"
    )
    assert "P01 TERMINATION OWNED reason=artifacts_secured" in result.stdout


def test_watchdog_unexpected_failure_leaves_label_for_launchd_restart(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    launchctl_log = tmp_path / "launchctl.log"
    fake_bin.mkdir()
    _executable(
        fake_bin / "launchctl",
        """
        #!/bin/sh
        printf '%s\n' "$*" >> "$FAKE_LAUNCHCTL_LOG"
        """,
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SC_REPO_ROOT": "/dev/null",
        "FAKE_LAUNCHCTL_LOG": str(launchctl_log),
    })
    result = subprocess.run(
        [
            "bash", str(LAUNCH), "--watch", "fixture",
            "/tmp/unused-p01-state.json", str(int(time.time())),
            "3600", "1.0", "com.semantic-continuity.p01.fixture",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode != 0
    assert not launchctl_log.exists()


def test_watchdog_attempts_pull_before_every_owned_termination() -> None:
    source = text(LAUNCH)
    completion_pull = source.index('bash "$PULL" "$name"')
    completion_terminate = source.index(
        'terminate_owned "$state"',
        completion_pull,
    )
    assert completion_pull < completion_terminate
    deadline = source.index("hard provider deadline reached")
    deadline_pull = source.index('bash "$PULL" "$name"', deadline)
    deadline_terminate = source.index(
        'terminate_owned "$state"',
        deadline_pull,
    )
    assert deadline_pull < deadline_terminate
    assert "pkill -INT" not in source
    assert "SIGINT" not in source
    assert "uv run python src/pod.py terminate" in source


def test_puller_is_receipt_driven_verifies_all_bytes_and_preserves_errors() -> None:
    source = text(PULL)
    assert "precision_probe_p01_pod_receipt_v1" in source
    assert 'terminal not in {"PASS","ERROR"}' in source
    assert 'actual != expected' in source
    assert "hashlib.sha256(data).hexdigest()" in source
    assert 'python3 "$PACKAGER" verify' in source
    assert "refusing different existing destination" in source
    assert "_QUARANTINE_precision_probe_p01_unreceipted_" in source
    assert "terminal receipt absent or invalid; completeness unverified" in source
    assert 'rsync -az --partial -e "$RSYNC_SSH"' in source
    assert '"root@$IP:$REMOTE_REPO/results/precision_probe_p01/"' in source
    assert "recovery_raw_count" in source
    assert "P01 PULL VERIFIED" in source


def _executable(path: Path, source: str) -> None:
    path.write_text(textwrap.dedent(source).lstrip())
    path.chmod(0o755)


def test_puller_installs_receipt_bound_error_without_runner_completion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote-repo"
    fake_bin = tmp_path / "bin"
    result_rel = Path("results/precision_probe_p01")
    receipt_rel = result_rel / (
        "precision-probe-p01-receipt_Qwen3-30B-A3B-Instruct-2507_"
        "20260712T070000Z.json"
    )
    partial_rel = result_rel / "run" / ".scratch" / "recovery.json"
    (root / "src").mkdir(parents=True)
    (root / "scripts/historical/coherent_canary_v12_e01/treatment_packaging").mkdir(
        parents=True
    )
    (root / "scripts/historical/coherent_canary_v12_e01/treatment_packaging/artifact_packager.py").write_text(
        "raise SystemExit('no package should be verified in this fixture')\n"
    )
    (root / ".pod_fake_state.json").write_text('{"id":"fake"}\n')
    fake_bin.mkdir()

    partial = remote / partial_rel
    partial.parent.mkdir(parents=True)
    partial.write_text('{"partial":true}\n')
    data = partial.read_bytes()
    receipt = {
        "schema": "precision_probe_p01_pod_receipt_v1",
        "protocol_id": "precision-probe-p01",
        "completed_at_utc": "2026-07-12T07:00:00+00:00",
        "expected_commit": "a" * 40,
        "observed_commit": "a" * 40,
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
        "runner_exit": 124,
        "runner_completion_status": None,
        "runner_completion_path": None,
        "packaging_exit": 1,
        "lossless_package_count": 0,
        "outcome_package_count": 0,
        "recovery_raw_count": 1,
        "terminal_status": "ERROR",
        "formal_v12_decision_eligible": False,
        "v12_reentry_authorized": False,
        "component_reuse_does_not_inherit_v12_eligibility": True,
        "provider": {
            "pod_name": "fake",
            "provider_pod_id": "pod-fake",
            "hourly_cost_usd": 1.39,
            "provider_clock_started_epoch": 1,
            "provider_wall_cap_seconds": 7200,
            "scientific_extension_cap_seconds": 7200,
            "operational_interrupt_elapsed_seconds": 7020,
            "operational_interrupt_epoch": 7021,
            "max_cost_usd": 4.0,
            "elapsed_seconds_at_receipt": 100,
        },
        "artifacts": [{
            "path": partial_rel.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }],
    }
    remote_receipt = remote / receipt_rel
    remote_receipt.parent.mkdir(parents=True, exist_ok=True)
    remote_receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n")

    _executable(
        fake_bin / "uv",
        """
        #!/bin/sh
        printf '%s\n' '{"publicIp":"127.0.0.1","portMappings":{"22":"2222"}}'
        """,
    )
    _executable(
        fake_bin / "ssh",
        f"""
        #!/bin/sh
        case "$*" in
          *"cat /workspace/exp/p01_current_receipt.txt"*)
            printf '%s\n' '{receipt_rel.as_posix()}';;
        esac
        exit 0
        """,
    )
    _executable(
        fake_bin / "rsync",
        r"""
        #!/usr/bin/env python3
        import os, pathlib, shutil, sys
        args=sys.argv[1:]
        remote=pathlib.Path(os.environ["FAKE_REMOTE_REPO"])
        destination=pathlib.Path(args[-1])
        files_arg=next((item for item in args if item.startswith("--files-from=")),None)
        if files_arg:
            destination.mkdir(parents=True,exist_ok=True)
            for value in pathlib.Path(files_arg.split("=",1)[1]).read_text().splitlines():
                source=remote / value
                target=destination / value
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
        else:
            source_arg=args[-2]
            if source_arg.endswith("/workspace/exp/job.log"):
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_text("partial job log\n")
            else:
                relative=source_arg.split(":",1)[1].split(
                    "/workspace/repo_precision_probe_p01/",1)[1]
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(remote / relative,destination)
        """,
    )
    _executable(fake_bin / "git", "#!/bin/sh\nexit 0\n")

    env = os.environ.copy()
    env.update({
        "PATH": f"{fake_bin}:{env['PATH']}",
        "SC_REPO_ROOT": str(root),
        "FAKE_REMOTE_REPO": str(remote),
    })
    result = subprocess.run(
        ["bash", str(PULL), "fake"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (root / partial_rel).read_bytes() == data
    assert (root / receipt_rel).is_file()
    assert "terminal_status=ERROR" in result.stdout
