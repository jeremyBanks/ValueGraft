"""Bounded provider lifecycle for the powered-v13 Stage-T canary.

The module owns allocation/admission/retry policy while the separately audited
``powered_v13_watchdog`` owns provider-clock cleanup.  All provider, SSH, and
process operations are injected; importing this module cannot allocate a Pod.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import importlib
import ipaddress
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error

from powered_v13_import_audit import (
    CONTRACT_SCHEMA,
    DESIGN_ID,
    FORBIDDEN_INVENTORY_PATHS,
    REPORT_SCHEMA as IMPORT_REPORT_SCHEMA,
    audit_stage_t_imports,
)
from powered_v13_watchdog import (
    DELETE_LEAD_SECONDS,
    MAX_PROVIDER_SECONDS,
    build_record,
    read_record,
    write_record_exclusive,
)


SCHEMA = "coherent-state-powered-successor-v13-stage-t-lifecycle-v1"
HARVEST_SCHEMA = "coherent-state-powered-successor-v13-stage-t-harvest-v1"
JOB_TERMINAL_SCHEMA = (
    "coherent-state-powered-successor-v13-stage-t-job-terminal-v1")
JOB_TERMINAL_PUBLISHER = (
    "import json,os,sys; "
    "terminal,state_path,rc_text,state,batch=sys.argv[1:]; "
    "rc=int(rc_text); "
    "assert state in {'COMPLETE','DEAD'}; "
    "assert (state=='COMPLETE' and rc==0) or "
    "(state=='DEAD' and rc!=0)\n"
    "def write(path,value):\n"
    " raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\\n').encode();\n"
    " tmp=path+'.tmp'; fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);\n"
    " handle=os.fdopen(fd,'wb'); handle.write(raw); handle.flush(); os.fsync(handle.fileno()); handle.close();\n"
    " os.replace(tmp,path); directory=os.open(os.path.dirname(path),os.O_RDONLY); os.fsync(directory); os.close(directory)\n"
    f"write(terminal,{{'schema':{JOB_TERMINAL_SCHEMA!r},"
    "'status':state,'returncode':rc,'primary_batch_id':batch}); "
    "write(state_path,{'state':state,'returncode':rc})"
)
MAX_ALLOCATION_ATTEMPTS = 2
MAX_EVENTS = 64
MAX_STATE_BYTES = 128 * 1024
MAX_HARVEST_FILES = 4096
MAX_HARVEST_BYTES = 2 * 1024 * 1024 * 1024
SSH_ADMISSION_TIMEOUT_SECONDS = 420
SYNC_TIMEOUT_SECONDS = 1800
LAUNCH_TIMEOUT_SECONDS = 60
TERMINAL_WAIT_SECONDS = 3400
JOB_PATH = "scripts/run_powered_v13_stage_t.py"
HARVEST_ROOTS = ("logs", "partials", "receipts", "results")
REMOTE_ROOT = "/workspace/powered-v13-stage-t"
REMOTE_REPO = f"{REMOTE_ROOT}/repo"
REMOTE_ARTIFACTS = f"{REMOTE_ROOT}/artifacts"
REMOTE_EXTERNAL = f"{REMOTE_ROOT}/external"
REMOTE_SECRETS = f"{REMOTE_ROOT}/secrets"
REMOTE_LAUNCH_RECEIPTS = f"{REMOTE_ROOT}/launch-receipts"
REMOTE_VENV = f"{REMOTE_ROOT}/venv"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAGE_T_RECEIPT_BASENAME = "powered-v13-technical-canary-launch-receipt.json"
EXPECTED_GPU_NAME = "NVIDIA A100 80GB PCIe"
EXPECTED_GPU_MEMORY_MIB = 81920
MINIMUM_NVIDIA_DRIVER = (580, 65, 6)
PROVIDER_CUDA_FILTER = "13.0"
POD_STATE_TOKEN = "{pod_state}"
LIFECYCLE_STATE_TOKEN = "{lifecycle_state}"
ATTEMPT_DIRECTORY_TOKEN = "{attempt_directory}"
_HEX_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_POD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{1,127}")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9][A-Za-z0-9-]{1,127}")
_DRIVER_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+")


class V13LifecycleError(RuntimeError):
    """A Stage-T provider lifecycle invariant failed closed."""


class NoCapacity(V13LifecycleError):
    """Provider positively reported no allocation."""


class AllocationAmbiguous(V13LifecycleError):
    """A create call ended without positive no-allocation or a Pod identity."""


class PodNotFound(V13LifecycleError):
    """Direct provider lookup positively returned 404."""


class AdmissionRejected(V13LifecycleError):
    """The allocated host failed the exact secure-A100 admission."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13LifecycleError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13LifecycleError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_exclusive(path: Path, raw: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def replace_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = canonical_json_bytes(dict(value)) + b"\n"
    _require(len(raw) <= MAX_STATE_BYTES, "lifecycle state exceeds byte bound")
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _strict_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(),
             f"{label} is absent or symlinked")
    raw = path.read_bytes()
    _require(0 < len(raw) <= MAX_STATE_BYTES, f"{label} byte bound differs")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise V13LifecycleError(f"{label} is not JSON: {exc}") from exc
    _require(isinstance(value, dict)
             and raw == canonical_json_bytes(value) + b"\n",
             f"{label} is not canonical newline-terminated JSON")
    return value, raw


def _safe_repo_path(value: object, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is empty")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute() and pure.as_posix() == value and pure.parts
             and all(part not in {"", ".", ".."} for part in pure.parts)
             and pure.parts[0] != ".git", f"{label} is unsafe")
    return value


def _exact_money(value: object, label: str, *, positive: bool = False) -> str:
    _require(isinstance(value, str), f"{label} is not an exact decimal string")
    try:
        observed = Decimal(value)
    except InvalidOperation as exc:
        raise V13LifecycleError(f"{label} is not decimal") from exc
    _require(observed.is_finite() and (observed > 0 if positive else observed >= 0),
             f"{label} is invalid")
    canonical = "0" if observed == 0 else format(observed.normalize(), "f")
    _require(canonical == value, f"{label} is not canonical fixed-point")
    return value


@dataclass(frozen=True)
class VerifiedRelease:
    authorization_commit: str
    manifest_path: str
    receipt_path: Path
    receipt_sha256: str
    import_report_path: Path
    import_report_sha256: str
    sync_paths: tuple[str, ...]
    job_path: str = JOB_PATH


@dataclass(frozen=True)
class Allocation:
    response: Mapping[str, Any]
    state_bytes: bytes

    @property
    def pod_id(self) -> str:
        value = self.response.get("id")
        _require(isinstance(value, str) and _POD_ID.fullmatch(value) is not None,
                 "allocated Pod ID is invalid")
        return value


class Provider(Protocol):
    def safe_snapshot(self) -> Mapping[str, Any]: ...
    def create_secure_a100(self) -> Allocation: ...
    def get_pod(self, pod_id: str) -> Mapping[str, Any]: ...
    def active_pod_ids(self) -> Sequence[str]: ...
    def delete_pod(self, pod_id: str) -> None: ...


class Transport(Protocol):
    def admit(self, allocation: Allocation, *, timeout_seconds: int) -> Mapping[str, Any]: ...
    def sync(self, allocation: Allocation, *, repo: Path,
             relative_paths: Sequence[str], external_files: Sequence[Path],
             timeout_seconds: int) -> Mapping[str, Any]: ...
    def launch_detached(self, allocation: Allocation, *, job_path: str,
                        authorization_commit: str,
                        timeout_seconds: int) -> Mapping[str, Any]: ...


class WatcherSupervisor(Protocol):
    def start(self, *, record_path: Path, pod_state_path: Path,
              lifecycle_state_path: Path) -> object: ...
    def request_cleanup(self, handle: object, *, reason: str) -> None: ...
    def wait_terminal(self, handle: object, *, timeout_seconds: int) -> Mapping[str, Any]: ...


class WatcherProcessBackend(Protocol):
    """OS-owned process seam; production may use launchd/systemd."""
    def start_watch(self, *, record_path: Path, pod_state_path: Path,
                    lifecycle_state_path: Path) -> object: ...
    def start_emergency(self, *, record_path: Path, pod_state_path: Path,
                        lifecycle_state_path: Path) -> object: ...
    def request_cleanup(self, *, lifecycle_state_path: Path,
                        reason: str) -> None: ...
    def wait(self, handle: object, *, timeout_seconds: int) -> int: ...
    def read_record(self, record_path: Path) -> Mapping[str, Any]: ...


class RecoveringWatcherSupervisor:
    """Restart ordinary watcher failure through the emergency-cleanup entry."""

    def __init__(self, backend: WatcherProcessBackend):
        self.backend = backend

    @staticmethod
    def _terminal(value: Mapping[str, Any]) -> bool:
        return value.get("status") in {
            "TERMINATED_HARVESTED", "TERMINATED_HARVEST_FAILED",
        }

    def start(self, *, record_path: Path, pod_state_path: Path,
              lifecycle_state_path: Path) -> object:
        return {
            "record_path": Path(record_path),
            "pod_state_path": Path(pod_state_path),
            "lifecycle_state_path": Path(lifecycle_state_path),
            "ordinary": self.backend.start_watch(
                record_path=Path(record_path),
                pod_state_path=Path(pod_state_path),
                lifecycle_state_path=Path(lifecycle_state_path)),
        }

    def request_cleanup(self, handle: object, *, reason: str) -> None:
        _require(isinstance(handle, Mapping), "watcher handle differs")
        self.backend.request_cleanup(
            lifecycle_state_path=Path(handle["lifecycle_state_path"]),
            reason=reason)

    def wait_terminal(self, handle: object, *, timeout_seconds: int) -> Mapping[str, Any]:
        _require(isinstance(handle, Mapping), "watcher handle differs")
        record_path = Path(handle["record_path"])
        pod_state_path = Path(handle["pod_state_path"])
        lifecycle_state_path = Path(handle["lifecycle_state_path"])
        self.backend.wait(handle["ordinary"], timeout_seconds=timeout_seconds)
        record = self.backend.read_record(record_path)
        if self._terminal(record):
            return deepcopy(dict(record))
        emergency = self.backend.start_emergency(
            record_path=record_path, pod_state_path=pod_state_path,
            lifecycle_state_path=lifecycle_state_path)
        self.backend.wait(emergency, timeout_seconds=timeout_seconds)
        recovered = self.backend.read_record(record_path)
        _require(self._terminal(recovered),
                 "emergency cleanup did not terminalize watchdog")
        return deepcopy(dict(recovered))


class RunPodProvider:
    """Narrow production adapter around the repository's credential-owning pod.py."""

    def __init__(
        self, *, state_path: Path, module: Any | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        reconcile_wait_seconds: int = 30,
    ):
        self.state_path = Path(state_path).resolve()
        self._baseline_active_ids: tuple[str, ...] | None = None
        _require(type(reconcile_wait_seconds) is int
                 and 1 <= reconcile_wait_seconds <= 60,
                 "ambiguous-create reconciliation wait is invalid")
        self.monotonic = monotonic
        self.sleep = sleep
        self.reconcile_wait_seconds = reconcile_wait_seconds
        self._enforce_provider_environment()
        self.module = module or importlib.reload(importlib.import_module("pod"))
        key_override = os.environ.get("SC_RUNPOD_KEY_PATH")
        if key_override:
            key_path = Path(key_override)
            _require(key_path.is_absolute() and key_path.is_file()
                     and not key_path.is_symlink(),
                     "absolute RunPod API key path differs")
            self.module.KEY_PATH = key_path

    def _enforce_provider_environment(self) -> None:
        """Pin provider-side filters immediately before every create call."""
        os.environ["SC_POD_STATE"] = str(self.state_path)
        os.environ["SC_POD_CLOUD"] = "SECURE"
        os.environ["SC_POD_ALLOWED_CUDA"] = PROVIDER_CUDA_FILTER
        os.environ.pop("SC_POD_SPOT", None)

    @staticmethod
    def _provider_money(value: object, label: str) -> str:
        _require(not isinstance(value, bool)
                 and isinstance(value, (int, float, str)),
                 f"{label} is not numeric")
        try:
            observed = Decimal(str(value))
        except InvalidOperation as exc:
            raise V13LifecycleError(f"{label} is not decimal") from exc
        _require(observed.is_finite() and observed >= 0,
                 f"{label} is invalid")
        return "0" if observed == 0 else format(observed.normalize(), "f")

    def safe_snapshot(self) -> Mapping[str, Any]:
        account = self.module.gql(
            "query { myself { clientBalance spendLimit } }")
        try:
            myself = account["data"]["myself"]
        except (KeyError, TypeError) as exc:
            raise V13LifecycleError("provider account snapshot differs") from exc
        snapshot = validate_provider_snapshot({
            "balance_usd": self._provider_money(
                myself.get("clientBalance"), "provider balance"),
            "spend_limit_usd": self._provider_money(
                myself.get("spendLimit"), "provider spend limit"),
            "active_pod_ids": list(self.active_pod_ids()),
        })
        self._baseline_active_ids = tuple(snapshot["active_pod_ids"])
        return snapshot

    def _attempt_evidence(self) -> tuple[Path, dict[str, Any]]:
        _require(self._baseline_active_ids is not None,
                 "provider create requires a complete preflight inventory")
        nonce = secrets.token_hex(12)
        name = f"powered-v13-stage-t-{nonce}"
        path = self.state_path.with_name(
            f"{self.state_path.name}.attempt-{nonce}.json")
        value = {
            "schema": "coherent-state-powered-successor-v13-create-recovery-v1",
            "attempt_nonce": nonce, "provider_name": name,
            "baseline_active_pod_ids": list(self._baseline_active_ids),
            "status": "PREPARED", "observed_active_pod_ids": None,
            "attributed_pod_id": None, "delete_requested": False,
            "per_pod_404_observed": False,
            "active_inventory_absent_observed": False,
        }
        write_bytes_exclusive(path, canonical_json_bytes(value) + b"\n")
        os.environ["SC_POD_NAME"] = name
        return path, value

    def _reconcile_ambiguous_create(
        self, *, evidence_path: Path, evidence: dict[str, Any],
        response: Mapping[str, Any] | None = None,
    ) -> None:
        baseline = set(evidence["baseline_active_pod_ids"])
        deadline = self.monotonic() + self.reconcile_wait_seconds
        polls = 0
        while True:
            try:
                active = list(self.active_pod_ids())
            except BaseException as exc:
                evidence.update({
                    "status": "RECONCILIATION_FAILED",
                    "reconciliation_error_type": type(exc).__name__,
                    "inventory_poll_count": polls,
                })
                replace_json(evidence_path, evidence)
                raise AllocationAmbiguous(
                    "post-create inventory is unreconciled; retry forbidden") from exc
            polls += 1
            evidence["observed_active_pod_ids"] = active
            evidence["inventory_poll_count"] = polls
            if not baseline.issubset(set(active)):
                evidence["status"] = "BASELINE_INVENTORY_CHANGED_UNRECONCILED"
                replace_json(evidence_path, evidence)
                raise AllocationAmbiguous(
                    "post-create inventory lost baseline Pods; retry forbidden")
            new_ids = sorted(set(active) - baseline)
            if new_ids or self.monotonic() >= deadline:
                break
            self.sleep(min(2, max(0, deadline - self.monotonic())))
        evidence["new_pod_ids"] = new_ids
        if not new_ids:
            evidence["status"] = "NO_NEW_POD_AMBIGUOUS"
            replace_json(evidence_path, evidence)
            raise AllocationAmbiguous(
                "ambiguous create exposed no Pod after bounded polling; retry forbidden")

        inspections: list[dict[str, Any]] = []
        attributable: list[tuple[str, str]] = []
        for pod_id in new_ids:
            try:
                pod = self.get_pod(pod_id)
                rate = self._provider_money(
                    pod.get("costPerHr"), "recovered Pod rate")
                valid = bool(
                    Decimal(rate) > 0
                    and pod.get("name") == evidence["provider_name"]
                    and pod.get("cloudType") == "SECURE"
                    and type(pod.get("gpuCount")) is int
                    and pod.get("gpuCount") == 1)
                inspections.append({
                    "pod_id": pod_id, "nonce_name_matches":
                        pod.get("name") == evidence["provider_name"],
                    "secure_one_gpu": pod.get("cloudType") == "SECURE"
                        and type(pod.get("gpuCount")) is int
                        and pod.get("gpuCount") == 1,
                    "canonical_cost_per_hr_usd": rate,
                    "attributable": valid,
                })
                if valid:
                    attributable.append((pod_id, rate))
            except BaseException as exc:
                inspections.append({
                    "pod_id": pod_id, "attributable": False,
                    "inspection_error_type": type(exc).__name__,
                })
        evidence["new_pod_inspections"] = inspections
        evidence["attributed_pod_ids"] = [pod_id for pod_id, _rate in attributable]
        if not attributable:
            evidence["status"] = "NO_NONCE_MATCHING_POD_UNRECONCILED"
            replace_json(evidence_path, evidence)
            raise AllocationAmbiguous(
                "new Pods did not match exact attempt nonce; retry forbidden")
        response_agrees = None
        if response is not None:
            try:
                response_rate = self._provider_money(
                    response.get("costPerHr"), "create response rate")
                response_agrees = any(
                    pod_id == response.get("id") and rate == response_rate
                    for pod_id, rate in attributable)
            except BaseException:
                response_agrees = False
        evidence.update({
            "status": "ATTRIBUTED_CLEANUP_STARTED",
            "attributed_pod_id": attributable[0][0],
            "canonical_cost_per_hr_usd": attributable[0][1],
            "response_agrees": response_agrees,
            "response_sha256": (
                sha256_bytes(canonical_json_bytes(dict(response)))
                if response is not None else None),
        })
        persistence_error: BaseException | None = None
        try:
            replace_json(evidence_path, evidence)
        except BaseException as exc:
            persistence_error = exc
        try:
            deleted: list[str] = []
            direct_404: list[str] = []
            for pod_id, _rate in attributable:
                self.delete_pod(pod_id)
                deleted.append(pod_id)
                try:
                    self.get_pod(pod_id)
                except PodNotFound:
                    direct_404.append(pod_id)
            evidence["delete_requested"] = bool(deleted)
            evidence["deleted_pod_ids"] = deleted
            evidence["direct_404_pod_ids"] = direct_404
            evidence["per_pod_404_observed"] = (
                direct_404 == [pod_id for pod_id, _rate in attributable])
            final_active = list(self.active_pod_ids())
            evidence["final_active_pod_ids"] = final_active
            evidence["active_inventory_absent_observed"] = all(
                pod_id not in final_active for pod_id, _rate in attributable)
            _require(evidence["per_pod_404_observed"] is True
                     and evidence["active_inventory_absent_observed"] is True,
                     "recovered Pod deletion lacks dual absence")
        except BaseException as exc:
            evidence["status"] = "ATTRIBUTED_CLEANUP_FAILED"
            evidence["cleanup_error_type"] = type(exc).__name__
            try:
                replace_json(evidence_path, evidence)
            except BaseException:
                pass
            raise AllocationAmbiguous(
                "attributable ambiguous Pod cleanup failed; retry forbidden") from exc
        evidence["status"] = "ATTRIBUTED_DELETED"
        if persistence_error is not None:
            evidence["predelete_evidence_error_type"] = type(
                persistence_error).__name__
        try:
            replace_json(evidence_path, evidence)
        except BaseException as exc:
            raise AllocationAmbiguous(
                "ambiguous Pod was deleted but recovery evidence failed; "
                "retry forbidden") from exc
        raise AllocationAmbiguous(
            "ambiguous create Pod was deleted and reconciled; retry forbidden")

    def create_secure_a100(self) -> Allocation:
        # Do this at call time as well as construction time: another in-process
        # caller must not be able to weaken the paid allocation filter.
        self._enforce_provider_environment()
        evidence_path, evidence = self._attempt_evidence()
        try:
            response = self.module.create(EXPECTED_GPU_NAME)
        except urllib.error.HTTPError as exc:
            if exc.code == 500:
                evidence["status"] = "PROVIDER_CONFIRMED_NO_CAPACITY"
                replace_json(evidence_path, evidence)
                raise NoCapacity("provider returned explicit no-capacity") from exc
            self._reconcile_ambiguous_create(
                evidence_path=evidence_path, evidence=evidence)
        except BaseException as exc:
            if isinstance(exc, (NoCapacity, AllocationAmbiguous)):
                raise
            self._reconcile_ambiguous_create(
                evidence_path=evidence_path, evidence=evidence)
        if (not isinstance(response, Mapping)
                or not isinstance(response.get("id"), str)
                or not response.get("id")):
            self._reconcile_ambiguous_create(
                evidence_path=evidence_path, evidence=evidence)
        try:
            state_bytes = self.state_path.read_bytes()
        except OSError as exc:
            self._reconcile_ambiguous_create(
                evidence_path=evidence_path, evidence=evidence,
                response=response)
        evidence["status"] = "EXACT_RESPONSE_AND_STATE_CAPTURED"
        evidence["attributed_pod_id"] = response.get("id")
        evidence["response_sha256"] = sha256_bytes(
            canonical_json_bytes(dict(response)))
        evidence["state_sha256"] = sha256_bytes(state_bytes)
        replace_json(evidence_path, evidence)
        return Allocation(response=deepcopy(dict(response)), state_bytes=state_bytes)

    def get_pod(self, pod_id: str) -> Mapping[str, Any]:
        try:
            value = self.module.api("GET", f"/pods/{pod_id}")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise PodNotFound(pod_id) from exc
            raise
        _require(isinstance(value, Mapping) and value.get("id") == pod_id,
                 "provider returned a different Pod")
        return value

    def active_pod_ids(self) -> Sequence[str]:
        value = self.module.api("GET", "/pods")
        _require(isinstance(value, list),
                 "provider active inventory is not a complete list")
        ids = [row.get("id") for row in value if isinstance(row, Mapping)]
        _require(len(ids) == len(value) and all(isinstance(item, str) for item in ids)
                 and len(ids) == len(set(ids)),
                 "provider active inventory differs")
        return sorted(ids)

    def delete_pod(self, pod_id: str) -> None:
        try:
            self.module.api("DELETE", f"/pods/{pod_id}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise


class OpenSshTransport:
    """Exact RunPod SSH, sparse-checkout, bootstrap, sync, and detach path."""

    def __init__(
        self, *, provider: Provider, ssh_key: Path, hf_token: Path,
        authorization_commit: str, primary_batch_id: str,
        manifest_path: str, receipt_sha256: str,
        import_report_path: str, import_report_sha256: str,
        repository_url: str = "https://github.com/jeremyBanks/ValueGraft.git",
        run_command: Callable[..., Any] = subprocess.run,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.provider = provider
        self.ssh_key = Path(ssh_key).expanduser().resolve()
        self.hf_token = Path(hf_token).expanduser().resolve()
        _require(self.ssh_key.is_file() and not self.ssh_key.is_symlink(),
                 "RunPod SSH key is absent or symlinked")
        _require(self.hf_token.is_file() and not self.hf_token.is_symlink()
                 and self.hf_token.stat().st_size > 0,
                 "Hugging Face token is empty, absent, or symlinked")
        self.model_download_auth_mode = "token_file"
        _require(repository_url.startswith("https://github.com/")
                 and not any(character.isspace() for character in repository_url),
                 "Stage-T repository URL is unsafe")
        self.repository_url = repository_url
        _require(_HEX_COMMIT.fullmatch(authorization_commit) is not None,
                 "transport authorization commit is malformed")
        self.authorization_commit = authorization_commit
        self.manifest_path = _safe_repo_path(
            manifest_path, "transport Stage-T manifest path")
        self.import_report_path = _safe_repo_path(
            import_report_path, "transport import-report path")
        _require(_SHA256.fullmatch(receipt_sha256) is not None
                 and _SHA256.fullmatch(import_report_sha256) is not None,
                 "transport release evidence hash is malformed")
        self.receipt_sha256 = receipt_sha256
        self.import_report_sha256 = import_report_sha256
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",
                              primary_batch_id) is not None
                 and len(primary_batch_id.encode("utf-8")) <= 128,
                 "transport primary batch ID is unsafe")
        self.primary_batch_id = primary_batch_id
        self.run_command = run_command
        self.monotonic = monotonic
        self.sleep = sleep
        self._endpoints: dict[str, tuple[str, int]] = {}

    def _run(self, command: Sequence[str], *, timeout_seconds: int,
             cwd: Path | None = None) -> Any:
        try:
            result = self.run_command(
                list(command), cwd=cwd, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False, timeout=timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise V13LifecycleError(
                f"bounded command failed to complete: {command[0]}") from exc
        _require(type(getattr(result, "returncode", None)) is int,
                 "bounded command returned no status")
        if result.returncode != 0:
            raise V13LifecycleError(
                f"bounded command failed ({command[0]}, {result.returncode})")
        return result

    def _endpoint(self, pod_id: str, *, timeout_seconds: int) -> tuple[str, int]:
        if pod_id in self._endpoints:
            return self._endpoints[pod_id]
        deadline = self.monotonic() + timeout_seconds
        while self.monotonic() < deadline:
            pod = self.provider.get_pod(pod_id)
            host = pod.get("publicIp")
            mappings = pod.get("portMappings")
            port_value = mappings.get("22") if isinstance(mappings, Mapping) else None
            try:
                address = ipaddress.ip_address(host) if isinstance(host, str) else None
                port = int(port_value)
            except (ValueError, TypeError):
                address = None
                port = 0
            if address is not None and 1 <= port <= 65535:
                endpoint = (str(address), port)
                self._endpoints[pod_id] = endpoint
                return endpoint
            self.sleep(min(5, max(0, deadline - self.monotonic())))
        raise AdmissionRejected("bounded SSH endpoint wait expired")

    def _ssh_transport(self, endpoint: tuple[str, int]) -> list[str]:
        host, port = endpoint
        return [
            "ssh", "-i", str(self.ssh_key), "-p", str(port),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=4",
        ]

    def _ssh(self, endpoint: tuple[str, int], remote: str) -> list[str]:
        return [*self._ssh_transport(endpoint), "-n",
                f"root@{endpoint[0]}", remote]

    def admit(self, allocation: Allocation, *, timeout_seconds: int) -> Mapping[str, Any]:
        endpoint = self._endpoint(
            allocation.pod_id, timeout_seconds=timeout_seconds)
        query = (
            "nvidia-smi --query-gpu=name,uuid,driver_version,memory.total "
            "--format=csv,noheader,nounits"
        )
        result = self._run(
            self._ssh(endpoint, query), timeout_seconds=min(90, timeout_seconds))
        rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
        _require(len(rows) == 1, "SSH admission did not expose exactly one GPU")
        fields = [field.strip() for field in rows[0].split(",")]
        _require(len(fields) == 4, "SSH admission GPU row is malformed")
        name, uuid, driver, memory_text = fields
        _require(memory_text.isdigit(), "SSH admission GPU memory is malformed")
        cuda_probe = (
            "python3 -c \"import torch; "
            "assert torch.cuda.is_available(); "
            "assert torch.cuda.device_count() == 1; "
            "print(torch.cuda.get_device_name(0))\""
        )
        cuda = self._run(
            self._ssh(endpoint, cuda_probe),
            timeout_seconds=min(90, timeout_seconds))
        torch_gpu_name = cuda.stdout.strip()
        _require(torch_gpu_name == EXPECTED_GPU_NAME,
                 "system-container Torch CUDA GPU differs")
        return {
            "gpu_name": name, "gpu_uuid": uuid, "driver_version": driver,
            "memory_mib": int(memory_text), "gpu_count": 1,
            "cuda_available": True, "torch_gpu_name": torch_gpu_name,
            "ssh_host": endpoint[0],
            "ssh_port": endpoint[1],
        }

    def sync(
        self, allocation: Allocation, *, repo: Path,
        relative_paths: Sequence[str], external_files: Sequence[Path],
        timeout_seconds: int,
    ) -> Mapping[str, Any]:
        endpoint = self._endpoints.get(allocation.pod_id)
        _require(endpoint is not None, "sync attempted before exact host admission")
        repo = Path(repo).resolve(strict=True)
        paths = [_safe_repo_path(path, "remote sync path")
                 for path in relative_paths]
        _require(paths == sorted(set(paths)) and JOB_PATH in paths
                 and self.import_report_path in paths,
                 "remote sync allowlist differs")
        _require(len(external_files) == 2,
                 "Stage-T external receipt/report set differs")
        deadline = self.monotonic() + timeout_seconds

        def remaining() -> int:
            value = int(deadline - self.monotonic())
            _require(value > 0, "bounded Stage-T sync/bootstrap deadline expired")
            return value

        commit = self._authorization_commit
        sparse = " ".join(shlex.quote(path) for path in paths)
        remote_prepare = (
            "set -euo pipefail; "
            f"test ! -e {shlex.quote(REMOTE_ROOT)}; "
            f"mkdir -p {shlex.quote(REMOTE_ROOT)}; "
            f"git clone --quiet --filter=blob:none --no-checkout --sparse -- "
            f"{shlex.quote(self.repository_url)} {shlex.quote(REMOTE_REPO + '.tmp')}; "
            f"git -C {shlex.quote(REMOTE_REPO + '.tmp')} sparse-checkout init --no-cone; "
            f"git -C {shlex.quote(REMOTE_REPO + '.tmp')} sparse-checkout set --no-cone -- {sparse}; "
            f"git -C {shlex.quote(REMOTE_REPO + '.tmp')} checkout --quiet --detach {shlex.quote(commit)}; "
            f"mv {shlex.quote(REMOTE_REPO + '.tmp')} {shlex.quote(REMOTE_REPO)}; "
            f"mkdir -p {shlex.quote(REMOTE_EXTERNAL)} "
            f"{shlex.quote(REMOTE_SECRETS)} {shlex.quote(REMOTE_ARTIFACTS)}"
        )
        self._run(self._ssh(endpoint, remote_prepare),
                  timeout_seconds=remaining())
        rsync_shell = shlex.join(self._ssh_transport(endpoint))
        self._run([
            "rsync", "-az", "--checksum", "--relative", "-e", rsync_shell,
            *paths, f"root@{endpoint[0]}:{REMOTE_REPO}/",
        ], cwd=repo, timeout_seconds=remaining())
        receipt = Path(external_files[0]).resolve(strict=True)
        report = Path(external_files[1]).resolve(strict=True)
        _require(receipt.is_file() and not receipt.is_symlink()
                 and file_sha256(receipt) == self.receipt_sha256,
                 "external Stage-T receipt differs")
        _require(report == (repo / self.import_report_path).resolve(strict=True)
                 and report.is_file() and not report.is_symlink()
                 and file_sha256(report) == self.import_report_sha256,
                 "inventoried Stage-T import-report authority differs")
        self._run([
            "rsync", "-az", "--checksum", "-e", rsync_shell,
            str(receipt),
            f"root@{endpoint[0]}:{REMOTE_EXTERNAL}/{STAGE_T_RECEIPT_BASENAME}",
        ], timeout_seconds=remaining())
        remote_verify = shlex.join([
            "python3", f"{REMOTE_REPO}/scripts/run_powered_v13_stage_t_lifecycle.py",
            "verify-release", "--repo", REMOTE_REPO,
            "--authorization-commit", commit,
            "--manifest", self.manifest_path,
            "--receipt", f"{REMOTE_EXTERNAL}/{STAGE_T_RECEIPT_BASENAME}",
            "--receipt-sha256", self.receipt_sha256,
            "--import-report", f"{REMOTE_REPO}/{self.import_report_path}",
            "--import-report-sha256", self.import_report_sha256,
        ])
        verified = self._run(
            self._ssh(endpoint,
                "set -euo pipefail; "
                f"cd {shlex.quote(REMOTE_REPO)}; "
                f"test \"$(git rev-parse HEAD)\" = {shlex.quote(commit)}; "
                "test -z \"$(git branch --show-current)\"; "
                "test -z \"$(git status --porcelain --untracked-files=all)\"; "
                f"PYTHONPATH=src {remote_verify}"),
            timeout_seconds=remaining())
        _require('"status": "PASS"' in verified.stdout,
                 "remote exact Stage-T release verification did not pass")
        self._run([
            "rsync", "-az", "--checksum", "-e", rsync_shell,
            str(self.hf_token),
            f"root@{endpoint[0]}:{REMOTE_SECRETS}/hf-token",
        ], timeout_seconds=remaining())
        dependencies = " ".join(shlex.quote(value) for value in (
            "torch==2.12.1", "transformers==5.0.0", "accelerate==1.14.0",
            "huggingface-hub==1.22.0", "safetensors==0.8.0",
            "sentencepiece==0.2.1", "tokenizers==0.22.2",
        ))
        environment_check = (
            "import importlib.metadata as m,platform,torch; "
            "e={'accelerate':'1.14.0','huggingface-hub':'1.22.0',"
            "'safetensors':'0.8.0','sentencepiece':'0.2.1',"
            "'tokenizers':'0.22.2','torch':'2.12.1',"
            "'transformers':'5.0.0'}; "
            "assert platform.python_version()=='3.12.11'; "
            "assert {k:m.version(k) for k in e}==e; "
            "assert torch.__version__=='2.12.1+cu130'; "
            "assert torch.version.cuda=='13.0' and torch.cuda.is_available(); "
            "assert torch.cuda.device_count()==1; "
            f"assert torch.cuda.get_device_name(0)=={EXPECTED_GPU_NAME!r}"
        )
        download_auth = (
            f"export HF_TOKEN=\"$(tr -d '[:space:]' < "
            f"{shlex.quote(REMOTE_SECRETS + '/hf-token')})\"; ")
        token_chmod = f"chmod 600 {shlex.quote(REMOTE_SECRETS + '/hf-token')}; "
        fresh_receipt_code = (
            "import json,sys; from pathlib import Path; "
            "from powered_v13_release import (STAGE_T_RECEIPT_BASENAME,"
            "create_stage_t_launch_receipt,sha256_bytes,verify_stage_t_checkout); "
            "repo,receipt_dir,commit,manifest=sys.argv[1:]; "
            "document=create_stage_t_launch_receipt(Path(repo),Path(receipt_dir),"
            "authorization_commit=commit,manifest_path=manifest); "
            "evidence=verify_stage_t_checkout(Path(repo),"
            "authorization_commit=commit,manifest_path=manifest,"
            "receipt_directory=Path(receipt_dir)); "
            "assert evidence.get('status')=='PASS' and "
            "evidence.get('detached_head')==commit and "
            "evidence.get('clean_tree') is True; "
            "path=Path(receipt_dir)/STAGE_T_RECEIPT_BASENAME; "
            "print(json.dumps({'fresh_receipt_sha256':sha256_bytes(path.read_bytes()),"
            "'fresh_receipt_created_utc':document['created_utc']},"
            "sort_keys=True,separators=(',',':')))"
        )
        remote_bootstrap = (
            "set -euo pipefail; "
            f"test \"$(git -C {shlex.quote(REMOTE_REPO)} rev-parse HEAD)\" = {shlex.quote(commit)}; "
            f"test -z \"$(git -C {shlex.quote(REMOTE_REPO)} branch --show-current)\"; "
            f"test -z \"$(git -C {shlex.quote(REMOTE_REPO)} status --porcelain --untracked-files=all)\"; "
            f"{token_chmod}"
            "python3 -m pip install -q --disable-pip-version-check --upgrade pip 'uv==0.9.18'; "
            "uv python install 3.12.11; "
            f"uv venv --python 3.12.11 {shlex.quote(REMOTE_VENV)}; "
            f"uv pip install --python {shlex.quote(REMOTE_VENV + '/bin/python')} {dependencies}; "
            f"test \"$({shlex.quote(REMOTE_VENV + '/bin/python')} -c "
            "'import platform; print(platform.python_version())')\" = 3.12.11; "
            "test \"$(uv --version)\" = 'uv 0.9.18'; "
            f"{shlex.quote(REMOTE_VENV + '/bin/python')} -c "
            f"{shlex.quote(environment_check)}; "
            f"{download_auth}"
            "export HF_HOME=/workspace/hf-cache HF_HUB_DISABLE_PROGRESS_BARS=1; "
            f"snapshot=$({shlex.quote(REMOTE_VENV + '/bin/python')} -c "
            f"{shlex.quote('from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=' + repr(MODEL_ID) + ', revision=' + repr(MODEL_REVISION) + ', local_files_only=False))')}); "
            f"test \"${{snapshot##*/}}\" = {shlex.quote(MODEL_REVISION)}; "
            f"test \"$(git -C {shlex.quote(REMOTE_REPO)} rev-parse HEAD)\" = {shlex.quote(commit)}; "
            f"test -z \"$(git -C {shlex.quote(REMOTE_REPO)} status --porcelain --untracked-files=all)\"; "
            f"test ! -e {shlex.quote(REMOTE_LAUNCH_RECEIPTS)}; "
            f"PYTHONPATH={shlex.quote(REMOTE_REPO + '/src')} "
            f"{shlex.quote(REMOTE_VENV + '/bin/python')} -c "
            f"{shlex.quote(fresh_receipt_code)} {shlex.quote(REMOTE_REPO)} "
            f"{shlex.quote(REMOTE_LAUNCH_RECEIPTS)} {shlex.quote(commit)} "
            f"{shlex.quote(self.manifest_path)}; "
            f"mkdir -p {shlex.quote(REMOTE_ARTIFACTS + '/receipts')}; "
            f"cp -n {shlex.quote(REMOTE_LAUNCH_RECEIPTS + '/' + STAGE_T_RECEIPT_BASENAME)} "
            f"{shlex.quote(REMOTE_ARTIFACTS + '/receipts/subject-launch-receipt.json')}; "
            f"test \"$(sha256sum {shlex.quote(REMOTE_LAUNCH_RECEIPTS + '/' + STAGE_T_RECEIPT_BASENAME)} | cut -d' ' -f1)\" = "
            f"\"$(sha256sum {shlex.quote(REMOTE_ARTIFACTS + '/receipts/subject-launch-receipt.json')} | cut -d' ' -f1)\""
        )
        bootstrapped = self._run(self._ssh(endpoint, remote_bootstrap),
                                 timeout_seconds=remaining())
        lines = [line for line in bootstrapped.stdout.splitlines() if line.strip()]
        _require(bool(lines), "fresh Stage-T receipt evidence is absent")
        try:
            fresh_receipt = json.loads(lines[-1])
        except json.JSONDecodeError as exc:
            raise V13LifecycleError(
                "fresh Stage-T receipt evidence is malformed") from exc
        _require(isinstance(fresh_receipt, Mapping)
                 and set(fresh_receipt) == {
                     "fresh_receipt_sha256", "fresh_receipt_created_utc"}
                 and _SHA256.fullmatch(str(
                     fresh_receipt.get("fresh_receipt_sha256"))) is not None
                 and fresh_receipt["fresh_receipt_sha256"] != self.receipt_sha256
                 and isinstance(fresh_receipt.get("fresh_receipt_created_utc"), str),
                 "fresh Stage-T receipt hash/time binding differs")
        self.fresh_receipt_sha256 = fresh_receipt["fresh_receipt_sha256"]
        self.fresh_receipt_created_utc = fresh_receipt[
            "fresh_receipt_created_utc"]
        return {
            "status": "PASS", "relative_paths": paths,
            "external_file_count": 1,
            "release_receipt_sha256": self.receipt_sha256,
            "fresh_launch_receipt_sha256": self.fresh_receipt_sha256,
            "fresh_launch_receipt_created_utc": self.fresh_receipt_created_utc,
            "inventoried_import_report_sha256": self.import_report_sha256,
            "remote_repo": REMOTE_REPO, "detached_head": commit,
            "python": "3.12.11", "uv": "0.9.18",
            "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "model_download_auth_mode": self.model_download_auth_mode,
        }

    @property
    def _authorization_commit(self) -> str:
        value = getattr(self, "authorization_commit", None)
        _require(isinstance(value, str) and _HEX_COMMIT.fullmatch(value) is not None,
                 "transport authorization commit is absent")
        return value

    def launch_detached(
        self, allocation: Allocation, *, job_path: str,
        authorization_commit: str, timeout_seconds: int,
    ) -> Mapping[str, Any]:
        endpoint = self._endpoints.get(allocation.pod_id)
        _require(endpoint is not None and job_path == JOB_PATH,
                 "detached launch identity differs")
        _require(authorization_commit == self.authorization_commit,
                 "detached launch authorization commit differs")
        state = f"{REMOTE_ARTIFACTS}/partials/job-state.json"
        log = f"{REMOTE_ARTIFACTS}/logs/job.log"
        start_receipt = f"{REMOTE_ARTIFACTS}/receipts/job-start.json"
        terminal_receipt = f"{REMOTE_ARTIFACTS}/receipts/job-terminal.json"
        scientific_command = shlex.join([
            REMOTE_VENV + "/bin/python", "-u", job_path,
            "--repo", REMOTE_REPO,
            "--receipt-directory", REMOTE_LAUNCH_RECEIPTS,
            "--output-parent", REMOTE_ARTIFACTS + "/results",
            "--primary-batch-id", self.primary_batch_id,
        ])
        completion_check = (
            "import json,sys; "
            "r=json.load(open(sys.argv[1]+'/RUN_COMPLETE.json')); "
            "s=json.load(open(sys.argv[1]+'/SUPERVISOR.json')); "
            "assert r.get('status')=='PASS' and "
            "r.get('primary_batch_id')==sys.argv[2]; "
            "assert s.get('status')=='WORKER_EXITED_ZERO' and "
            "s.get('worker_returncode')==0"
        )
        wrapper = (
            f"{scientific_command}; status=$?; "
            f"run_count=$(find {shlex.quote(REMOTE_ARTIFACTS + '/results')} -mindepth 1 -maxdepth 1 -type d -name 'powered-v13-stage-t_*' | wc -l | tr -d ' '); "
            f"run_dir=$(find {shlex.quote(REMOTE_ARTIFACTS + '/results')} -mindepth 1 -maxdepth 1 -type d -name 'powered-v13-stage-t_*' -print | head -1); "
            "state=DEAD; "
            f"if [ \"$status\" -eq 0 ] && [ \"$run_count\" -eq 1 ] && "
            "[ -f \"$run_dir/RUN_COMPLETE.json\" ] && "
            "[ -f \"$run_dir/SUPERVISOR.json\" ] && "
            f"{shlex.quote(REMOTE_VENV + '/bin/python')} -c {shlex.quote(completion_check)} "
            f"\"$run_dir\" {shlex.quote(self.primary_batch_id)}; "
            "then state=COMPLETE; else [ \"$status\" -ne 0 ] || status=22; fi; "
            f"{shlex.quote(REMOTE_VENV + '/bin/python')} -c "
            f"{shlex.quote(JOB_TERMINAL_PUBLISHER)} "
            f"{shlex.quote(terminal_receipt)} "
            f"{shlex.quote(state)} \"$status\" \"$state\" "
            f"{shlex.quote(self.primary_batch_id)}; "
            "exit \"$status\""
        )
        job_auth = (
            f"export HF_TOKEN=\"$(tr -d '[:space:]' < "
            f"{shlex.quote(REMOTE_SECRETS + '/hf-token')})\"; ")
        remote = (
            "set -euo pipefail; "
            f"cd {shlex.quote(REMOTE_REPO)}; "
            f"test \"$(git rev-parse HEAD)\" = {shlex.quote(authorization_commit)}; "
            "test -z \"$(git branch --show-current)\"; "
            "test -z \"$(git status --porcelain --untracked-files=all)\"; "
            f"mkdir -p {' '.join(shlex.quote(REMOTE_ARTIFACTS + '/' + root) for root in HARVEST_ROOTS)}; "
            f"printf '{{\"state\":\"RUNNING\"}}\\n' > {shlex.quote(state)}; "
            f"printf '{{\"status\":\"RUNNING\"}}\\n' > {shlex.quote(start_receipt)}; "
            f"{job_auth}"
            "export CUDA_VISIBLE_DEVICES=0 HF_HOME=/workspace/hf-cache "
            "HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 "
            "TOKENIZERS_PARALLELISM=false PYTHONPATH=src; "
            f"nohup /bin/bash -c {shlex.quote(wrapper)} "
            f"</dev/null >{shlex.quote(log)} 2>&1 & pid=$!; "
            f"printf '%s\\n' \"$pid\" > {shlex.quote(REMOTE_ROOT + '/job.pid')}; "
            "disown \"$pid\"; printf '%s\\n' \"$pid\""
        )
        launched = self._run(self._ssh(endpoint, remote),
                             timeout_seconds=timeout_seconds)
        pid_text = launched.stdout.strip()
        _require(pid_text.isdigit() and int(pid_text) > 1,
                 "detached launch returned no child PID")
        progress = (
            "set -euo pipefail; "
            f"pid=$(cat {shlex.quote(REMOTE_ROOT + '/job.pid')}); "
            "for i in $(seq 1 30); do "
            f"if kill -0 \"$pid\" 2>/dev/null && test -s {shlex.quote(log)} "
            f"&& test -s {shlex.quote(state)}; then echo PASS; exit 0; fi; "
            "sleep 1; done; exit 21"
        )
        checked = self._run(self._ssh(endpoint, progress),
                            timeout_seconds=min(45, timeout_seconds))
        _require(checked.stdout.strip() == "PASS",
                 "detached child made no bounded log/state progress")
        return {
            "status": "PASS", "pipe_eof": True, "child_alive": True,
            "log_progress": True, "pid": int(pid_text), "job_path": job_path,
            "remote_state_path": state, "remote_log_path": log,
        }


class LaunchdWatcherBackend:
    """OS-owned launchd+caffeinate execution of the existing watchdog CLI."""

    def __init__(
        self, *, repo: Path, python: Path | None = None,
        runpod_key_path: Path,
        launchctl: str = "/bin/launchctl",
        caffeinate: str = "/usr/bin/caffeinate",
        run_command: Callable[..., Any] = subprocess.run,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.repo = Path(repo).resolve(strict=True)
        self.python = Path(python or sys.executable).resolve(strict=True)
        self.runpod_key_path = Path(runpod_key_path).resolve(strict=True)
        _require(self.runpod_key_path.is_file()
                 and not self.runpod_key_path.is_symlink(),
                 "RunPod API key path is absent or symlinked")
        self.launchctl = launchctl
        self.caffeinate = caffeinate
        self.run_command = run_command
        self.monotonic = monotonic
        self.sleep = sleep
        self.domain = f"gui/{os.getuid()}"

    def _run(self, command: Sequence[str], *, timeout: int = 30) -> Any:
        try:
            return self.run_command(
                list(command), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, check=False, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise V13LifecycleError(
                f"launchd command did not complete: {command[0]}") from exc

    def _label(self, record_path: Path, mode: str) -> str:
        record = read_record(record_path)
        pod = re.sub(r"[^A-Za-z0-9-]", "-", record["pod_id"])[-40:]
        suffix = sha256_bytes(str(Path(record_path).resolve()).encode())[:12]
        return f"com.openai.powered-v13.stage-t.{pod}.{suffix}.{mode}"

    def _print(self, label: str) -> Any:
        return self._run([self.launchctl, "print", f"{self.domain}/{label}"])

    def _remove(self, label: str) -> None:
        observed = self._print(label)
        if observed.returncode != 0:
            return
        removed = self._run([
            self.launchctl, "bootout", f"{self.domain}/{label}"])
        if removed.returncode != 0:
            removed = self._run([self.launchctl, "remove", label])
        _require(removed.returncode == 0 and self._print(label).returncode != 0,
                 "launchd watchdog label could not be positively evicted")

    def _start(
        self, *, mode: str, record_path: Path, pod_state_path: Path,
        lifecycle_state_path: Path,
    ) -> Mapping[str, Any]:
        record_path = Path(record_path).resolve(strict=True)
        pod_state_path = Path(pod_state_path).resolve(strict=True)
        lifecycle_state_path = Path(lifecycle_state_path).resolve(strict=True)
        record = read_record(record_path)
        _require(file_sha256(pod_state_path) == record["pod_state_sha256"],
                 "watchdog Pod-state hash differs")
        label = self._label(record_path, mode)
        _require(self._print(label).returncode != 0,
                 "launchd watchdog label already exists")
        log = record_path.with_name(
            "watchdog.log" if mode == "ordinary" else "watchdog-emergency.log")
        subcommand = "watch" if mode == "ordinary" else "emergency-cleanup"
        watcher = self.repo / "scripts/run_powered_v13_stage_t_watchdog.py"
        command = [
            self.launchctl, "submit", "-l", label, "-o", str(log),
            "-e", str(log), "--", "/usr/bin/env",
            f"HOME={Path.home()}", f"PATH={os.environ.get('PATH', '')}",
            f"SC_REPO_ROOT={self.repo}",
            f"SC_RUNPOD_KEY_PATH={self.runpod_key_path}",
            self.caffeinate, "-dimsu",
            str(self.python), str(watcher), subcommand,
            "--state", str(pod_state_path), "--record", str(record_path),
        ]
        submitted = self._run(command)
        _require(submitted.returncode == 0,
                 "provider-clock watchdog launchctl submission failed")
        self.sleep(1)
        status = self._print(label)
        _require(status.returncode == 0 and re.search(
            r"(?m)^\s*state = running\s*$", status.stdout) is not None,
            "provider-clock watchdog is not launchd-running")
        return {
            "label": label, "record_path": record_path,
            "pod_state_path": pod_state_path,
            "lifecycle_state_path": lifecycle_state_path, "mode": mode,
        }

    def start_watch(self, *, record_path: Path, pod_state_path: Path,
                    lifecycle_state_path: Path) -> object:
        return self._start(
            mode="ordinary", record_path=record_path,
            pod_state_path=pod_state_path,
            lifecycle_state_path=lifecycle_state_path)

    def start_emergency(self, *, record_path: Path, pod_state_path: Path,
                        lifecycle_state_path: Path) -> object:
        self._remove(self._label(record_path, "ordinary"))
        return self._start(
            mode="emergency", record_path=record_path,
            pod_state_path=pod_state_path,
            lifecycle_state_path=lifecycle_state_path)

    def request_cleanup(self, *, lifecycle_state_path: Path,
                        reason: str) -> None:
        value = json.loads(Path(lifecycle_state_path).read_bytes())
        _require(isinstance(value, Mapping)
                 and value.get("status") == "FORCE_CLEANUP"
                 and value.get("force_cleanup_reason") == reason,
                 "lifecycle cleanup signal differs")

    def wait(self, handle: object, *, timeout_seconds: int) -> int:
        _require(isinstance(handle, Mapping), "launchd handle differs")
        label = str(handle["label"])
        record_path = Path(handle["record_path"])
        deadline = self.monotonic() + timeout_seconds
        while self.monotonic() < deadline:
            record = read_record(record_path)
            if record.get("status") in {
                "TERMINATED_HARVESTED", "TERMINATED_HARVEST_FAILED",
            }:
                self._remove(label)
                return 0
            status = self._print(label)
            if status.returncode != 0 or re.search(
                    r"(?m)^\s*state = running\s*$", status.stdout) is None:
                return 1
            self.sleep(min(1, max(0, deadline - self.monotonic())))
        self._remove(label)
        return 124

    def read_record(self, record_path: Path) -> Mapping[str, Any]:
        return read_record(record_path)


def verify_release_binding(
    *, repo: Path, expected_authorization_commit: str, manifest_path: str,
    receipt_path: Path, expected_receipt_sha256: str,
    import_report_path: Path, expected_import_report_sha256: str,
    verify_checkout: Callable[..., Mapping[str, Any]],
) -> VerifiedRelease:
    """Bind checkout, receipt, import report, and the exact sync allowlist."""

    repo = Path(repo).resolve(strict=True)
    _require(_HEX_COMMIT.fullmatch(expected_authorization_commit) is not None,
             "expected Stage-T authorization commit is malformed")
    manifest_path = _safe_repo_path(manifest_path, "Stage-T manifest path")
    _require(_SHA256.fullmatch(expected_receipt_sha256) is not None
             and _SHA256.fullmatch(expected_import_report_sha256) is not None,
             "expected release/report hash is malformed")
    receipt, receipt_raw = _strict_json(receipt_path, "Stage-T launch receipt")
    _require(sha256_bytes(receipt_raw) == expected_receipt_sha256,
             "Stage-T launch receipt hash differs")
    _require(receipt.get("design_id") == DESIGN_ID
             and receipt.get("stage") == "TECHNICAL_CANARY"
             and receipt.get("authorization_commit") == expected_authorization_commit
             and receipt.get("manifest_path") == manifest_path,
             "Stage-T launch receipt binding differs")
    import_report_path = Path(import_report_path).resolve(strict=True)
    try:
        import_report_relative = import_report_path.relative_to(repo).as_posix()
    except ValueError as exc:
        raise V13LifecycleError(
            "Stage-T import report is not inside the exact checkout") from exc
    import_report_relative = _safe_repo_path(
        import_report_relative, "Stage-T import report path")
    report, report_raw = _strict_json(import_report_path, "Stage-T import report")
    _require(sha256_bytes(report_raw) == expected_import_report_sha256,
             "Stage-T import report file hash differs")
    claimed_report_hash = report.get("report_sha256")
    report_payload = dict(report)
    report_payload.pop("report_sha256", None)
    _require(report.get("schema") == IMPORT_REPORT_SCHEMA
             and report.get("design_id") == DESIGN_ID
             and report.get("stage") == "TECHNICAL_CANARY"
             and report.get("status") == "PASS"
             and report.get("semantic_n") == 0
             and report.get("production_entropy_requested") is False
             and claimed_report_hash == sha256_bytes(
                 canonical_json_bytes(report_payload)),
             "Stage-T import report binding differs")
    roots = report.get("execution_roots")
    _require(isinstance(roots, list) and JOB_PATH in roots,
             "Stage-T import report does not audit the exact job")

    evidence = verify_checkout(
        repo=repo,
        authorization_commit=expected_authorization_commit,
        manifest_path=manifest_path,
        receipt_directory=Path(receipt_path).parent,
    )
    _require(isinstance(evidence, Mapping) and evidence.get("status") == "PASS"
             and evidence.get("detached_head") == expected_authorization_commit,
             "Stage-T checkout verification failed")
    authorization = evidence.get("authorization")
    verified_receipt = evidence.get("receipt")
    _require(isinstance(authorization, Mapping)
             and authorization.get("authorization_commit") ==
             expected_authorization_commit
             and isinstance(verified_receipt, Mapping)
             and verified_receipt.get("receipt_sha256") == expected_receipt_sha256,
             "Stage-T verifier evidence differs")

    manifest, manifest_raw = _strict_json(repo / manifest_path, "Stage-T manifest")
    _require(sha256_bytes(manifest_raw) == authorization.get("manifest_sha256"),
             "Stage-T manifest hash differs from authorization")
    inventory = manifest.get("inventory")
    _require(isinstance(inventory, list) and bool(inventory),
             "Stage-T manifest inventory is empty")
    paths = [_safe_repo_path(row.get("path"), f"inventory {index} path")
             for index, row in enumerate(inventory)
             if isinstance(row, Mapping)]
    _require(len(paths) == len(inventory) and paths == sorted(set(paths)),
             "Stage-T manifest inventory paths differ")
    report_rows = [row for row in inventory
                   if isinstance(row, Mapping)
                   and row.get("path") == import_report_relative]
    _require(len(report_rows) == 1,
             "Stage-T import report is absent from manifest inventory")
    report_row = report_rows[0]
    _require(report_row.get("mode") == "100644"
             and type(report_row.get("bytes")) is int
             and report_row.get("bytes") == len(report_raw)
             and report_row.get("sha256") == expected_import_report_sha256,
             "Stage-T manifest does not bind exact import-report bytes")
    changed = authorization.get("changed_paths")
    _require(isinstance(changed, list), "authorization changed paths are absent")
    changed_paths = [_safe_repo_path(path, "authorization changed path")
                     for path in changed]
    sync_paths = tuple(sorted(set(paths + changed_paths)))
    forbidden = sorted(set(sync_paths) & set(FORBIDDEN_INVENTORY_PATHS))
    _require(not forbidden, f"production-only paths reached Stage T: {forbidden}")
    _require(JOB_PATH in paths, "exact Stage-T job is absent from parent inventory")
    contract_path = report.get("contract_path")
    _require(contract_path in paths, "import-audit contract is absent from inventory")
    roots = report.get("execution_roots")
    _require(isinstance(roots, list), "import-audit execution roots are absent")
    recomputed_report = audit_stage_t_imports(
        repo, contract_path=Path(contract_path), execution_roots=roots)
    _require(report == recomputed_report
             and report_raw == canonical_json_bytes(recomputed_report) + b"\n",
             "Stage-T import report differs from exact-checkout recomputation")
    return VerifiedRelease(
        authorization_commit=expected_authorization_commit,
        manifest_path=manifest_path,
        receipt_path=Path(receipt_path).resolve(strict=True),
        receipt_sha256=expected_receipt_sha256,
        import_report_path=import_report_path,
        import_report_sha256=expected_import_report_sha256,
        sync_paths=sync_paths,
    )


def validate_admission(pod: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require(pod.get("cloudType") == "SECURE", "allocated host is not secure cloud")
    _require(type(pod.get("gpuCount")) is int and pod.get("gpuCount") == 1,
             "allocated host does not have one GPU")
    name = evidence.get("gpu_name")
    memory = evidence.get("memory_mib")
    _require(name == EXPECTED_GPU_NAME,
             "host GPU name differs from the exact A100 80GB PCIe contract")
    _require(type(memory) is int and memory == EXPECTED_GPU_MEMORY_MIB,
             "host GPU memory differs from the exact 81920 MiB contract")
    driver = evidence.get("driver_version")
    _require(isinstance(driver, str)
             and _DRIVER_VERSION.fullmatch(driver) is not None,
             "host NVIDIA driver version is malformed")
    driver_components = tuple(int(part) for part in driver.split("."))
    _require(driver_components >= MINIMUM_NVIDIA_DRIVER,
             "host NVIDIA driver is below 580.65.06")
    _require(evidence.get("cuda_available") is True
             and type(evidence.get("gpu_count")) is int
             and evidence.get("gpu_count") == 1
             and evidence.get("torch_gpu_name") == EXPECTED_GPU_NAME
             and isinstance(evidence.get("gpu_uuid"), str)
             and _GPU_UUID.fullmatch(evidence["gpu_uuid"]) is not None,
             "host one-GPU/CUDA/UUID admission differs")
    result = deepcopy(dict(evidence))
    result["driver_components"] = list(driver_components)
    return result


def validate_provider_snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == {
        "balance_usd", "spend_limit_usd", "active_pod_ids",
    }, "provider snapshot fields differ")
    _exact_money(value.get("balance_usd"), "provider balance")
    _exact_money(value.get("spend_limit_usd"), "provider spend limit")
    ids = value.get("active_pod_ids")
    _require(isinstance(ids, list) and ids == sorted(set(ids))
             and all(isinstance(item, str) and _POD_ID.fullmatch(item)
                     for item in ids), "provider active Pod inventory differs")
    return deepcopy(dict(value))


def job_probe_exit(*, lifecycle_status: str, remote_state: str | None) -> int:
    """Fixed watchdog exit contract without shell-dependent coercion."""
    if lifecycle_status in {"ALLOCATING", "WATCHDOG_STARTED", "ADMITTED", "SYNCED"}:
        return 0
    if lifecycle_status == "FORCE_CLEANUP":
        return 21
    _require(lifecycle_status == "JOB_STARTED", "job probe lifecycle status differs")
    _require(remote_state in {"RUNNING", "COMPLETE", "DEAD"},
             "remote job state is ambiguous")
    return {"RUNNING": 0, "COMPLETE": 20, "DEAD": 21}[remote_state]


def build_harvest_manifest(root: Path) -> dict[str, Any]:
    """Hash every bounded result/log/receipt/partial file after transport."""
    root = Path(root).resolve(strict=True)
    rows: list[dict[str, Any]] = []
    total = 0
    for category in HARVEST_ROOTS:
        directory = root / category
        _require(not directory.is_symlink() and directory.is_dir(),
                 f"harvest category is absent or symlinked: {category}")
        for path in sorted(directory.rglob("*")):
            _require(not path.is_symlink(), "harvest path contains a symlink")
            if path.is_dir():
                continue
            _require(path.is_file(), "harvest path is not regular")
            size = path.stat().st_size
            total += size
            _require(len(rows) < MAX_HARVEST_FILES
                     and total <= MAX_HARVEST_BYTES,
                     "harvest exceeds file/byte bound")
            rows.append({
                "category": category,
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": file_sha256(path),
            })
    _require(all(any(row["category"] == category for row in rows)
                 for category in HARVEST_ROOTS),
             "harvest omits a required artifact category")
    document = {
        "schema": HARVEST_SCHEMA,
        "design_id": DESIGN_ID,
        "file_count": len(rows),
        "total_bytes": total,
        "files": rows,
    }
    document["manifest_sha256"] = sha256_bytes(canonical_json_bytes(document))
    return document


class StageTLifecycle:
    """One-shot, at-most-two-attempt lifecycle with one admitted host."""

    def __init__(
        self, *, repo: Path, session_root: Path, release: VerifiedRelease,
        provider: Provider, transport: Transport, supervisor: WatcherSupervisor,
        clock: Callable[[], float], prior_stage_t_spend_usd: str,
        prior_stage_t_provider_seconds: int,
        job_probe_command: Sequence[str], harvest_command: Sequence[str],
    ):
        self.repo = Path(repo).resolve(strict=True)
        self.root = Path(session_root)
        self.release = release
        self.provider = provider
        self.transport = transport
        self.supervisor = supervisor
        self.clock = clock
        self.prior_spend = _exact_money(
            prior_stage_t_spend_usd, "prior Stage-T spend")
        self.accumulated_spend = Decimal(self.prior_spend)
        _require(type(prior_stage_t_provider_seconds) is int
                 and 0 <= prior_stage_t_provider_seconds <= MAX_PROVIDER_SECONDS,
                 "prior Stage-T provider seconds are invalid")
        self.accumulated_provider_seconds = prior_stage_t_provider_seconds
        self.job_probe_command = tuple(job_probe_command)
        self.harvest_command = tuple(harvest_command)
        _require(self.job_probe_command and self.harvest_command,
                 "watchdog commands are empty")
        _require(not self.root.exists(), "lifecycle session root already exists")
        self.root.mkdir(parents=True)
        self.state_path = self.root / "lifecycle.json"
        self.state: dict[str, Any] = {
            "schema": SCHEMA,
            "design_id": DESIGN_ID,
            "authorization_commit": release.authorization_commit,
            "status": "INITIALIZED",
            "attempts": [],
            "admitted_pod_id": None,
            "job_started": False,
            "observed_stage_t_spend_usd": self.prior_spend,
            "observed_stage_t_provider_seconds":
                self.accumulated_provider_seconds,
            "force_cleanup_reason": None,
            "events": [],
        }
        self._event("INITIALIZED", {})

    def _event(self, kind: str, evidence: Mapping[str, Any]) -> None:
        _require(len(self.state["attempts"]) <= MAX_ALLOCATION_ATTEMPTS
                 and len(self.state["events"]) < MAX_EVENTS,
                 "lifecycle retained state exceeds bound")
        self.state["events"].append({
            "epoch": int(self.clock()), "kind": kind,
            "evidence": deepcopy(dict(evidence)),
        })
        replace_json(self.state_path, self.state)

    def _set(self, status: str, kind: str, evidence: Mapping[str, Any], **changes: Any) -> None:
        self.state.update(changes)
        self.state["status"] = status
        self._event(kind, evidence)

    def _persist_allocation(self, allocation: Allocation, attempt: int,
                            started: int) -> tuple[Path, Path, Path]:
        directory = self.root / f"attempt-{attempt}"
        directory.mkdir()
        response_path = directory / "create-response.json"
        response_raw = canonical_json_bytes(dict(allocation.response)) + b"\n"
        write_bytes_exclusive(response_path, response_raw)
        state_path = directory / "pod-state.json"
        write_bytes_exclusive(state_path, bytes(allocation.state_bytes))
        try:
            state_doc = json.loads(allocation.state_bytes)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise V13LifecycleError("provider Pod state is not JSON") from exc
        pod_id = allocation.pod_id
        _require(isinstance(state_doc, Mapping) and state_doc.get("id") == pod_id
                 and state_doc.get("costPerHr") == allocation.response.get("costPerHr"),
                 "create response and Pod state differ")
        watchdog_path = directory / "watchdog.json"
        replacements = {
            POD_STATE_TOKEN: str(state_path.resolve()),
            LIFECYCLE_STATE_TOKEN: str(self.state_path.resolve()),
            ATTEMPT_DIRECTORY_TOKEN: str(directory.resolve()),
        }
        def bind(command: Sequence[str]) -> tuple[str, ...]:
            values: list[str] = []
            for value in command:
                for token, replacement in replacements.items():
                    value = value.replace(token, replacement)
                values.append(value)
            return tuple(values)
        job_probe_command = bind(self.job_probe_command)
        harvest_command = bind(self.harvest_command)
        _require(not ({POD_STATE_TOKEN, LIFECYCLE_STATE_TOKEN,
                       ATTEMPT_DIRECTORY_TOKEN}
                      & set(job_probe_command + harvest_command)),
                 "watchdog command placeholders were not bound")
        record = build_record(
            pod_id=pod_id,
            created_cost_per_hr_usd=str(allocation.response.get("costPerHr")),
            prior_stage_t_spend_usd=self.prior_spend,
            prior_stage_t_provider_seconds=
                self.accumulated_provider_seconds,
            provider_clock_started_epoch=started,
            pod_state_sha256=file_sha256(state_path),
            create_response_sha256=file_sha256(response_path),
            job_sha256=file_sha256(self.repo / self.release.job_path),
            release_receipt_sha256=self.release.receipt_sha256,
            job_probe_command=job_probe_command,
            harvest_command=harvest_command,
        )
        write_record_exclusive(watchdog_path, record)
        return response_path, state_path, watchdog_path

    def _charge_attempt(
        self, allocation: Allocation, *, started: int,
    ) -> dict[str, Any]:
        elapsed = max(0, int(self.clock()) - started)
        try:
            rate = Decimal(str(allocation.response.get("costPerHr")))
        except InvalidOperation as exc:
            raise V13LifecycleError("allocated rate is not decimal") from exc
        _require(rate.is_finite() and rate > 0, "allocated rate is invalid")
        self.accumulated_spend += rate * Decimal(elapsed) / Decimal(3600)
        self.accumulated_provider_seconds += elapsed
        value = ("0" if self.accumulated_spend == 0 else
                 format(self.accumulated_spend.normalize(), "f"))
        self.prior_spend = value
        self.state["observed_stage_t_spend_usd"] = value
        self.state["observed_stage_t_provider_seconds"] = (
            self.accumulated_provider_seconds)
        return {
            "attempt_provider_seconds": elapsed,
            "observed_stage_t_provider_seconds":
                self.accumulated_provider_seconds,
            "observed_stage_t_spend_usd": value,
        }

    def _require_attempt_provider_window(self, attempt: int) -> None:
        remaining = MAX_PROVIDER_SECONDS - self.accumulated_provider_seconds
        if remaining <= DELETE_LEAD_SECONDS:
            evidence = {
                "attempt": attempt,
                "observed_stage_t_provider_seconds":
                    self.accumulated_provider_seconds,
                "remaining_stage_t_provider_seconds": remaining,
                "delete_lead_seconds": DELETE_LEAD_SECONDS,
            }
            self._set("BLOCKED", "PROVIDER_SECONDS_EXHAUSTED", evidence)
            raise V13LifecycleError(
                "remaining Stage-T provider seconds cannot fund safe cleanup")

    def _positive_absence(self, pod_id: str) -> None:
        try:
            self.provider.get_pod(pod_id)
        except PodNotFound:
            pass
        else:
            raise V13LifecycleError("deleted Pod did not return direct 404")
        ids = list(self.provider.active_pod_ids())
        _require(pod_id not in ids and ids == sorted(set(ids)),
                 "deleted Pod remains in complete active inventory")

    def _direct_delete_before_watchdog(self, pod_id: str) -> None:
        """Own cleanup when allocation succeeded before supervision could start."""
        try:
            self.provider.delete_pod(pod_id)
        except PodNotFound:
            pass
        self._positive_absence(pod_id)

    def _cleanup(self, handle: object, *, pod_id: str, reason: str) -> Mapping[str, Any]:
        self._set("FORCE_CLEANUP", "CLEANUP_REQUESTED", {"reason": reason},
                  force_cleanup_reason=reason)
        self.supervisor.request_cleanup(handle, reason=reason)
        terminal = self.supervisor.wait_terminal(
            handle, timeout_seconds=TERMINAL_WAIT_SECONDS)
        _require(terminal.get("status") in {
            "TERMINATED_HARVESTED", "TERMINATED_HARVEST_FAILED",
        } and terminal.get("per_pod_404_observed") is True
            and terminal.get("active_inventory_absent_observed") is True,
            "watchdog did not prove dual deletion")
        self._positive_absence(pod_id)
        return terminal

    @staticmethod
    def _terminal_ok(terminal: Mapping[str, Any]) -> bool:
        return bool(terminal.get("status") in {
            "TERMINATED_HARVESTED", "TERMINATED_HARVEST_FAILED",
        } and terminal.get("per_pod_404_observed") is True
            and terminal.get("active_inventory_absent_observed") is True
            and isinstance(terminal.get("harvest"), Mapping))

    def run(self) -> dict[str, Any]:
        snapshot = validate_provider_snapshot(self.provider.safe_snapshot())
        _require(snapshot["active_pod_ids"] == [],
                 "Stage T requires zero active Pods before allocation")
        self._set("READY", "PROVIDER_PREFLIGHT", snapshot)
        for attempt in range(1, MAX_ALLOCATION_ATTEMPTS + 1):
            self._require_attempt_provider_window(attempt)
            self._set("ALLOCATING", "ALLOCATION_CLOCK_STARTED", {
                "attempt": attempt,
                "prior_stage_t_provider_seconds":
                    self.accumulated_provider_seconds,
            })
            started = int(self.clock())
            try:
                allocation = self.provider.create_secure_a100()
            except NoCapacity:
                self.state["attempts"].append({
                    "attempt": attempt, "clock_started_epoch": started,
                    "result": "NO_CAPACITY", "pod_id": None,
                })
                self._event("NO_CAPACITY", {"attempt": attempt})
                if attempt == MAX_ALLOCATION_ATTEMPTS:
                    self._set("NO_CAPACITY", "ATTEMPTS_EXHAUSTED", {})
                    return deepcopy(self.state)
                continue
            except BaseException as exc:
                self.state["attempts"].append({
                    "attempt": attempt, "clock_started_epoch": started,
                    "result": "ALLOCATION_AMBIGUOUS", "pod_id": None,
                })
                self._set("BLOCKED", "ALLOCATION_AMBIGUOUS", {
                    "attempt": attempt, "error_type": type(exc).__name__,
                })
                raise AllocationAmbiguous(
                    "provider allocation is ambiguous; retry forbidden") from exc

            pod_id = allocation.pod_id
            try:
                response_path, state_path, watchdog_path = self._persist_allocation(
                    allocation, attempt, started)
                handle = self.supervisor.start(
                    record_path=watchdog_path, pod_state_path=state_path,
                    lifecycle_state_path=self.state_path)
            except BaseException as exc:
                try:
                    self._direct_delete_before_watchdog(pod_id)
                except BaseException as cleanup_exc:
                    self._set("BLOCKED", "PRE_WATCHDOG_CLEANUP_FAILED", {
                        "attempt": attempt,
                        "error_type": type(cleanup_exc).__name__,
                    })
                    raise V13LifecycleError(
                        "pre-watchdog cleanup failed; retry forbidden") from cleanup_exc
                charge = self._charge_attempt(allocation, started=started)
                self.state["attempts"].append({
                    "attempt": attempt, "clock_started_epoch": started,
                    "result": "PRE_WATCHDOG_REJECTED_DELETED", "pod_id": pod_id,
                    **charge,
                })
                self._event("PRE_WATCHDOG_REJECTED_DELETED", {
                    "attempt": attempt, "error_type": type(exc).__name__,
                    **charge,
                })
                if attempt == MAX_ALLOCATION_ATTEMPTS:
                    self._set("BLOCKED", "ATTEMPTS_EXHAUSTED", {})
                    raise V13LifecycleError(
                        "pre-watchdog setup failed twice; allocation stopped") from exc
                continue
            self.state["attempts"].append({
                "attempt": attempt, "clock_started_epoch": started,
                "result": "ALLOCATED", "pod_id": pod_id,
                "create_response_sha256": file_sha256(response_path),
                "pod_state_sha256": file_sha256(state_path),
                "watchdog_path": watchdog_path.relative_to(self.root).as_posix(),
            })
            self._set("WATCHDOG_STARTED", "WATCHDOG_STARTED", {
                "attempt": attempt, "pod_id": pod_id,
            })
            try:
                admission = self.transport.admit(
                    allocation, timeout_seconds=SSH_ADMISSION_TIMEOUT_SECONDS)
                admission = validate_admission(allocation.response, admission)
            except BaseException as exc:
                terminal = self._cleanup(
                    handle, pod_id=pod_id, reason="admission_rejected")
                charge = self._charge_attempt(allocation, started=started)
                self.state["attempts"][-1]["result"] = "REJECTED_DELETED"
                self.state["attempts"][-1].update(charge)
                self._event("ADMISSION_REJECTED_DELETED", {
                    "attempt": attempt, "harvest_succeeded":
                    terminal["harvest"]["succeeded"],
                    "error_type": type(exc).__name__,
                    **charge,
                })
                if attempt == MAX_ALLOCATION_ATTEMPTS:
                    self._set("ADMISSION_REJECTED", "ATTEMPTS_EXHAUSTED", {})
                    return deepcopy(self.state)
                continue

            _require(self.state["admitted_pod_id"] is None,
                     "more than one Stage-T host was admitted")
            self._set("ADMITTED", "HOST_ADMITTED", admission,
                      admitted_pod_id=pod_id)
            try:
                sync_evidence = self.transport.sync(
                    allocation, repo=self.repo,
                    relative_paths=self.release.sync_paths,
                    external_files=(self.release.receipt_path,
                                    self.release.import_report_path),
                    timeout_seconds=SYNC_TIMEOUT_SECONDS)
                _require(sync_evidence.get("status") == "PASS"
                         and sync_evidence.get("relative_paths") ==
                         list(self.release.sync_paths),
                         "remote sync evidence differs")
                self._set("SYNCED", "BOUND_INPUTS_SYNCED", sync_evidence)
                launch = self.transport.launch_detached(
                    allocation, job_path=self.release.job_path,
                    authorization_commit=self.release.authorization_commit,
                    timeout_seconds=LAUNCH_TIMEOUT_SECONDS)
                _require(launch.get("status") == "PASS"
                         and launch.get("pipe_eof") is True
                         and launch.get("child_alive") is True,
                         "detached job launch evidence differs")
                self._set("JOB_STARTED", "JOB_STARTED", launch, job_started=True)
                terminal = self.supervisor.wait_terminal(
                    handle, timeout_seconds=TERMINAL_WAIT_SECONDS)
                _require(self._terminal_ok(terminal),
                         "job terminal lacks dual deletion proof")
            except BaseException as exc:
                # Once a host is admitted (and especially after job/model start),
                # no later allocation is permitted. Recover this Pod only.
                try:
                    terminal = self._cleanup(
                        handle, pod_id=pod_id, reason="post_admission_failure")
                except BaseException as cleanup_exc:
                    self._set("BLOCKED", "CLEANUP_FAILED", {
                        "pod_id": pod_id,
                        "error_type": type(cleanup_exc).__name__,
                    })
                    raise V13LifecycleError(
                        "post-admission cleanup failed; retry forbidden") from cleanup_exc
                charge = self._charge_attempt(allocation, started=started)
                self.state["attempts"][-1].update(charge)
                self._set("BLOCKED", "POST_ADMISSION_FAILURE", {
                    "pod_id": pod_id, "error_type": type(exc).__name__,
                    "cleanup_status": terminal["status"],
                    **charge,
                })
                raise V13LifecycleError(
                    "post-admission failure cleaned; retry forbidden") from exc
            self._positive_absence(pod_id)
            charge = self._charge_attempt(allocation, started=started)
            self.state["attempts"][-1].update(charge)
            successful_job = bool(
                terminal["status"] == "TERMINATED_HARVESTED"
                and terminal.get("termination_reason") == "job_complete"
                and terminal["harvest"].get("succeeded") is True)
            final = ("COMPLETE" if successful_job else
                     ("STOPPED" if terminal["status"] ==
                      "TERMINATED_HARVESTED" else "HARVEST_FAILED"))
            self._set(final, "NO_WARM_HOLD_TERMINAL", {
                "pod_id": pod_id,
                "termination_reason": terminal.get("termination_reason"),
                "harvest_succeeded": terminal["harvest"]["succeeded"],
                **charge,
            })
            return deepcopy(self.state)
        raise AssertionError("bounded attempt loop fell through")


__all__ = [
    "Allocation", "AllocationAmbiguous", "AdmissionRejected", "HARVEST_ROOTS",
    "JOB_PATH", "MAX_ALLOCATION_ATTEMPTS", "NoCapacity", "PodNotFound",
    "LaunchdWatcherBackend", "OpenSshTransport", "Provider",
    "RecoveringWatcherSupervisor", "RunPodProvider", "SCHEMA",
    "StageTLifecycle", "Transport", "V13LifecycleError", "WatcherProcessBackend",
    "VerifiedRelease", "WatcherSupervisor", "build_harvest_manifest",
    "canonical_json_bytes", "job_probe_exit", "sha256_bytes",
    "validate_admission", "validate_provider_snapshot", "verify_release_binding",
]
