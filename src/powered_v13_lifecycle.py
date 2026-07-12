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
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error

from powered_v13_import_audit import (
    CONTRACT_SCHEMA,
    DESIGN_ID,
    FORBIDDEN_INVENTORY_PATHS,
    REPORT_SCHEMA as IMPORT_REPORT_SCHEMA,
    audit_stage_t_imports,
)
from powered_v13_watchdog import build_record, write_record_exclusive


SCHEMA = "coherent-state-powered-successor-v13-stage-t-lifecycle-v1"
HARVEST_SCHEMA = "coherent-state-powered-successor-v13-stage-t-harvest-v1"
MAX_ALLOCATION_ATTEMPTS = 2
MAX_EVENTS = 64
MAX_STATE_BYTES = 128 * 1024
MAX_HARVEST_FILES = 4096
MAX_HARVEST_BYTES = 2 * 1024 * 1024 * 1024
SSH_ADMISSION_TIMEOUT_SECONDS = 420
SYNC_TIMEOUT_SECONDS = 300
LAUNCH_TIMEOUT_SECONDS = 60
TERMINAL_WAIT_SECONDS = 3400
JOB_PATH = "scripts/run_powered_v13_stage_t.py"
HARVEST_ROOTS = ("logs", "partials", "receipts", "results")
EXPECTED_GPU_NAME = "NVIDIA A100 80GB PCIe"
EXPECTED_GPU_MEMORY_MIB = 81920
MINIMUM_NVIDIA_DRIVER = (580, 65, 6)
PROVIDER_CUDA_FILTER = "13.0"
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
    def start(self, *, record_path: Path, state_path: Path) -> object: ...
    def request_cleanup(self, handle: object, *, reason: str) -> None: ...
    def wait_terminal(self, handle: object, *, timeout_seconds: int) -> Mapping[str, Any]: ...


class WatcherProcessBackend(Protocol):
    """OS-owned process seam; production may use launchd/systemd."""
    def start_watch(self, *, record_path: Path, state_path: Path) -> object: ...
    def start_emergency(self, *, record_path: Path, state_path: Path) -> object: ...
    def request_cleanup(self, *, state_path: Path, reason: str) -> None: ...
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

    def start(self, *, record_path: Path, state_path: Path) -> object:
        return {
            "record_path": Path(record_path), "state_path": Path(state_path),
            "ordinary": self.backend.start_watch(
                record_path=Path(record_path), state_path=Path(state_path)),
        }

    def request_cleanup(self, handle: object, *, reason: str) -> None:
        _require(isinstance(handle, Mapping), "watcher handle differs")
        self.backend.request_cleanup(
            state_path=Path(handle["state_path"]), reason=reason)

    def wait_terminal(self, handle: object, *, timeout_seconds: int) -> Mapping[str, Any]:
        _require(isinstance(handle, Mapping), "watcher handle differs")
        record_path = Path(handle["record_path"])
        state_path = Path(handle["state_path"])
        self.backend.wait(handle["ordinary"], timeout_seconds=timeout_seconds)
        record = self.backend.read_record(record_path)
        if self._terminal(record):
            return deepcopy(dict(record))
        emergency = self.backend.start_emergency(
            record_path=record_path, state_path=state_path)
        self.backend.wait(emergency, timeout_seconds=timeout_seconds)
        recovered = self.backend.read_record(record_path)
        _require(self._terminal(recovered),
                 "emergency cleanup did not terminalize watchdog")
        return deepcopy(dict(recovered))


class RunPodProvider:
    """Narrow production adapter around the repository's credential-owning pod.py."""

    def __init__(self, *, state_path: Path, module: Any | None = None):
        self.state_path = Path(state_path).resolve()
        self._enforce_provider_environment()
        self.module = module or importlib.reload(importlib.import_module("pod"))

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
        return validate_provider_snapshot({
            "balance_usd": self._provider_money(
                myself.get("clientBalance"), "provider balance"),
            "spend_limit_usd": self._provider_money(
                myself.get("spendLimit"), "provider spend limit"),
            "active_pod_ids": list(self.active_pod_ids()),
        })

    def create_secure_a100(self) -> Allocation:
        # Do this at call time as well as construction time: another in-process
        # caller must not be able to weaken the paid allocation filter.
        self._enforce_provider_environment()
        try:
            response = self.module.create(EXPECTED_GPU_NAME)
        except urllib.error.HTTPError as exc:
            if exc.code == 500:
                raise NoCapacity("provider returned explicit no-capacity") from exc
            raise AllocationAmbiguous("provider create HTTP result is ambiguous") from exc
        except BaseException as exc:
            raise AllocationAmbiguous("provider create transport is ambiguous") from exc
        _require(isinstance(response, Mapping), "provider create response differs")
        try:
            state_bytes = self.state_path.read_bytes()
        except OSError as exc:
            raise AllocationAmbiguous(
                "provider returned a Pod but state capture failed") from exc
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
        record = build_record(
            pod_id=pod_id,
            created_cost_per_hr_usd=str(allocation.response.get("costPerHr")),
            prior_stage_t_spend_usd=self.prior_spend,
            provider_clock_started_epoch=started,
            pod_state_sha256=file_sha256(state_path),
            create_response_sha256=file_sha256(response_path),
            job_sha256=file_sha256(self.repo / self.release.job_path),
            release_receipt_sha256=self.release.receipt_sha256,
            job_probe_command=self.job_probe_command,
            harvest_command=self.harvest_command,
        )
        write_record_exclusive(watchdog_path, record)
        return response_path, state_path, watchdog_path

    def _charge_attempt(self, allocation: Allocation, *, started: int) -> str:
        elapsed = max(0, int(self.clock()) - started)
        try:
            rate = Decimal(str(allocation.response.get("costPerHr")))
        except InvalidOperation as exc:
            raise V13LifecycleError("allocated rate is not decimal") from exc
        _require(rate.is_finite() and rate > 0, "allocated rate is invalid")
        self.accumulated_spend += rate * Decimal(elapsed) / Decimal(3600)
        value = ("0" if self.accumulated_spend == 0 else
                 format(self.accumulated_spend.normalize(), "f"))
        self.prior_spend = value
        self.state["observed_stage_t_spend_usd"] = value
        return value

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
            self._set("ALLOCATING", "ALLOCATION_CLOCK_STARTED", {"attempt": attempt})
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
                    record_path=watchdog_path, state_path=self.state_path)
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
                self._charge_attempt(allocation, started=started)
                self.state["attempts"].append({
                    "attempt": attempt, "clock_started_epoch": started,
                    "result": "PRE_WATCHDOG_REJECTED_DELETED", "pod_id": pod_id,
                })
                self._event("PRE_WATCHDOG_REJECTED_DELETED", {
                    "attempt": attempt, "error_type": type(exc).__name__,
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
                spend = self._charge_attempt(allocation, started=started)
                self.state["attempts"][-1]["result"] = "REJECTED_DELETED"
                self._event("ADMISSION_REJECTED_DELETED", {
                    "attempt": attempt, "harvest_succeeded":
                    terminal["harvest"]["succeeded"],
                    "error_type": type(exc).__name__,
                    "observed_stage_t_spend_usd": spend,
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
                self._charge_attempt(allocation, started=started)
                self._set("BLOCKED", "POST_ADMISSION_FAILURE", {
                    "pod_id": pod_id, "error_type": type(exc).__name__,
                    "cleanup_status": terminal["status"],
                })
                raise V13LifecycleError(
                    "post-admission failure cleaned; retry forbidden") from exc
            self._positive_absence(pod_id)
            self._charge_attempt(allocation, started=started)
            final = ("COMPLETE" if terminal["status"] ==
                     "TERMINATED_HARVESTED" else "HARVEST_FAILED")
            self._set(final, "NO_WARM_HOLD_TERMINAL", {
                "pod_id": pod_id,
                "termination_reason": terminal.get("termination_reason"),
                "harvest_succeeded": terminal["harvest"]["succeeded"],
            })
            return deepcopy(self.state)
        raise AssertionError("bounded attempt loop fell through")


__all__ = [
    "Allocation", "AllocationAmbiguous", "AdmissionRejected", "HARVEST_ROOTS",
    "JOB_PATH", "MAX_ALLOCATION_ATTEMPTS", "NoCapacity", "PodNotFound",
    "Provider", "RecoveringWatcherSupervisor", "RunPodProvider", "SCHEMA",
    "StageTLifecycle", "Transport", "V13LifecycleError", "WatcherProcessBackend",
    "VerifiedRelease", "WatcherSupervisor", "build_harvest_manifest",
    "canonical_json_bytes", "job_probe_exit", "sha256_bytes",
    "validate_admission", "validate_provider_snapshot", "verify_release_binding",
]
