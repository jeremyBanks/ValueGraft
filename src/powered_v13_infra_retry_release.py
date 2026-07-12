"""Pure-Git release boundary for powered-v13 Stage-T infrastructure retry 1.

The retry is an outer operational authorization only.  It binds the exhausted
Stage-T scientific payload, the deterministic provider-schema repair, the full
rejection evidence, one additional allocation, and no fallback.  It neither
imports nor changes the lifecycle/controller.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


DESIGN_ID = "coherent-state-powered-successor-v13"
STAGE = "TECHNICAL_CANARY_INFRA_RETRY_1"
MANIFEST_SCHEMA = "powered-v13-stage-t-infra-retry-manifest-v1"
CONTRACT_SCHEMA = "powered-v13-stage-t-infra-retry-inventory-contract-v1"
RECEIPT_SCHEMA = "powered-v13-stage-t-infra-retry-launch-receipt-v1"

AMENDMENT_PATH = (
    "data/coherent_state_powered_v13/stage-t-infra-retry-1-amendment.md"
)
CONTRACT_PATH = (
    "data/coherent_state_powered_v13/stage-t-infra-retry-1-inventory-contract.json"
)
RECEIPT_BASENAME = "powered-v13-stage-t-infra-retry-1-launch-receipt.json"
OLD_STATUS_LINE = b"**Status:** **DRAFT \xe2\x80\x94 NO PROVIDER ALLOCATION AUTHORIZED**\n"
NEW_STATUS_LINE = (
    b"**Status:** **INFRA_RETRY_AUTHORIZED \xe2\x80\x94 ONE EXTRA ALLOCATION; "
    b"NO FALLBACK; SEMANTIC N=0**\n"
)

ORIGINAL_AUTHORIZATION_COMMIT = "766db12ff12de52f5826238f7446435bef047ab1"
ORIGINAL_PREREGISTRATION_PATH = (
    "COHERENT-STATE-POWERED-SUCCESSOR-V13-PREREGISTRATION.md"
)
ORIGINAL_MANIFEST_PATH = (
    "results/coherent_state_powered_v13/releases/"
    "powered-v13-stage-t-manifest_20260712T2240Z.json"
)
ORIGINAL_MANIFEST_SHA256 = (
    "c774d65fa9bf0a0ab5f9bdbd9cc09ea43da04ba7d78f4ec9c5cbb07a583ae98f"
)
PROVIDER_PATCH_COMMIT = "79ab77f10d743e97153af8485532bcf5592092ef"
ONE_ATTEMPT_CONTROLLER_COMMIT = "ffc90ce6e39b7fd08ad3dd77cfb7fb51fe53f6d2"
INCIDENT_PATH = "notes/powered-v13-stage-t-admission-rejected-provider-schema.md"
INCIDENT_SHA256 = (
    "0388d036b1dcd51ea7ee8575e967751f626259c1576d2d4d15ef1af2879ed257"
)
FAILURE_ROOT = (
    "results/coherent_state_powered_v13/"
    "stage-t-admission-rejected_20260712T224225Z"
)

CARRY_IN_PROVIDER_SECONDS = 58
CARRY_IN_CONSERVATIVE_SPEND_USD = "0.02239444444444444444444444444"
CARRY_IN_OBSERVED_PROVIDER_DELTA_USD = "0.0197885805"
CUMULATIVE_PROVIDER_SECONDS_CAP = 3300
CUMULATIVE_SPEND_CAP_USD = "1.50"
MAX_ADDITIONAL_PROVIDER_SECONDS = 3242
MAX_ADDITIONAL_SPEND_USD = "1.251772222222222222222222222"
PINNED_COST_PER_HOUR_USD = "1.39"
CONSUMPTION_ROOT_RELATIVE = (
    ".local/state/valuegraft/powered-v13-stage-t-infra-retry-1-consumed"
)

FAILURE_EVIDENCE_PATHS = (
    INCIDENT_PATH,
    f"{FAILURE_ROOT}/session/attempt-1/create-response.json",
    f"{FAILURE_ROOT}/session/attempt-1/pod-state.json",
    f"{FAILURE_ROOT}/session/attempt-1/watchdog.json",
    f"{FAILURE_ROOT}/session/attempt-1/watchdog.log",
    f"{FAILURE_ROOT}/session/attempt-2/create-response.json",
    f"{FAILURE_ROOT}/session/attempt-2/pod-state.json",
    f"{FAILURE_ROOT}/session/attempt-2/watchdog.json",
    f"{FAILURE_ROOT}/session/attempt-2/watchdog.log",
    f"{FAILURE_ROOT}/session/lifecycle.json",
    f"{FAILURE_ROOT}/setup-receipt/"
    "powered-v13-technical-canary-launch-receipt.json",
)
CONTROLLER_PATHS = (
    "scripts/run_powered_v13_stage_t_lifecycle.py",
    "scripts/run_powered_v13_stage_t_watchdog.py",
    "src/pod.py",
    "src/powered_v13_import_audit.py",
    "src/powered_v13_lifecycle.py",
    "src/powered_v13_release.py",
    "src/powered_v13_watchdog.py",
)
FIXED_INVENTORY_PATHS = tuple(sorted((
    AMENDMENT_PATH,
    CONTRACT_PATH,
    ORIGINAL_MANIFEST_PATH,
    "scripts/run_powered_v13_stage_t_infra_retry_1.py",
    "src/powered_v13_infra_retry_entrypoint.py",
    "src/powered_v13_infra_retry_release.py",
    *CONTROLLER_PATHS,
    *FAILURE_EVIDENCE_PATHS,
)))

DEFAULT_MAX_RECEIPT_AGE_SECONDS = 300
DEFAULT_FUTURE_SKEW_SECONDS = 5
_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_PATH_RE = re.compile(r"[A-Za-z0-9._/-]+\Z")
_MANIFEST_PATH_RE = re.compile(
    r"results/coherent_state_powered_v13/releases/"
    r"powered-v13-stage-t-infra-retry-1-manifest_[0-9]{8}T[0-9]{4,6}Z\.json\Z"
)


class InfraRetryReleaseError(RuntimeError):
    """The infrastructure-retry release failed closed."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InfraRetryReleaseError(f"value is not canonical-JSON safe: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"
    env["GIT_LITERAL_PATHSPECS"] = "1"
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False,
        )
    except OSError as exc:
        raise InfraRetryReleaseError(f"git invocation failed: {exc}") from exc
    if check and result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise InfraRetryReleaseError(
            f"git {' '.join(args)} failed with {result.returncode}: {detail}"
        )
    return result


def _git_text(repo: Path, *args: str) -> str:
    return _git(repo, *args).stdout.decode("utf-8", "strict")


def _safe_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not _PATH_RE.fullmatch(value):
        raise InfraRetryReleaseError(f"{label} is not a safe repository-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in path.parts
    ) or path.parts[0] == ".git":
        raise InfraRetryReleaseError(f"{label} is not a normalized repository path")
    return value


def _exact_commit(repo: Path, value: Any, label: str) -> str:
    if not isinstance(value, str) or not _OID_RE.fullmatch(value):
        raise InfraRetryReleaseError(f"{label} is not an exact lowercase Git OID")
    resolved = _git_text(repo, "rev-parse", "--verify", f"{value}^{{commit}}").strip()
    if resolved != value:
        raise InfraRetryReleaseError(f"{label} does not resolve exactly")
    return value


def _require_ancestor(repo: Path, ancestor: str, descendant: str, label: str) -> None:
    result = _git(repo, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if result.returncode != 0:
        raise InfraRetryReleaseError(f"{label} is not an ancestor of static root")


def _tree_blob(repo: Path, commit: str, path: str) -> tuple[str, bytes]:
    path = _safe_path(path, "tree path")
    raw = _git(repo, "ls-tree", "-z", "--full-tree", commit, "--", path).stdout
    records = [record for record in raw.split(b"\0") if record]
    if len(records) != 1:
        raise InfraRetryReleaseError(f"{path} is absent or ambiguous in {commit}")
    try:
        header, raw_name = records[0].split(b"\t", 1)
        mode, kind, oid = header.decode("ascii").split(" ")
        name = raw_name.decode("utf-8", "strict")
    except (ValueError, UnicodeDecodeError) as exc:
        raise InfraRetryReleaseError(f"malformed tree entry for {path}") from exc
    if name != path or kind != "blob" or mode not in {"100644", "100755"}:
        raise InfraRetryReleaseError(f"{path} is not one regular tracked blob")
    return mode, _git(repo, "cat-file", "blob", oid).stdout


def _path_absent(repo: Path, commit: str, path: str) -> bool:
    return not any(
        record for record in _git(
            repo, "ls-tree", "-z", "--full-tree", commit, "--", path
        ).stdout.split(b"\0") if record
    )


def _parents(repo: Path, commit: str) -> list[str]:
    fields = _git_text(repo, "rev-list", "--parents", "-n", "1", commit).split()
    return fields[1:]


def _canonical_object(blob: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InfraRetryReleaseError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or blob != canonical_json_bytes(value) + b"\n":
        raise InfraRetryReleaseError(f"{label} is not canonical JSON plus LF")
    return value


def _contract_paths(repo: Path, root: str) -> tuple[str, ...]:
    _mode, blob = _tree_blob(repo, root, CONTRACT_PATH)
    contract = _canonical_object(blob, "retry inventory contract")
    if set(contract) != {"schema", "paths"} or contract.get("schema") != CONTRACT_SCHEMA:
        raise InfraRetryReleaseError("retry inventory contract schema/fields differ")
    paths = contract.get("paths")
    if not isinstance(paths, list) or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise InfraRetryReleaseError("retry inventory contract paths are not unique/sorted")
    normalized = tuple(_safe_path(path, "contract path") for path in paths)
    if normalized != FIXED_INVENTORY_PATHS:
        raise InfraRetryReleaseError("retry inventory contract differs from fixed paths")
    return normalized


def _inventory(repo: Path, root: str, paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        mode, blob = _tree_blob(repo, root, path)
        rows.append({
            "bytes": len(blob), "mode": mode, "path": path,
            "sha256": sha256_bytes(blob),
        })
    return rows


def _verify_original_authorization(repo: Path, static_root: str) -> dict[str, Any]:
    authorization = _exact_commit(
        repo, ORIGINAL_AUTHORIZATION_COMMIT, "original authorization commit"
    )
    _require_ancestor(repo, authorization, static_root, "original authorization")
    parents = _parents(repo, authorization)
    if len(parents) != 1:
        raise InfraRetryReleaseError("original authorization does not have one parent")
    parent = parents[0]
    raw_diff = _git(
        repo, "diff-tree", "--no-commit-id", "--name-status", "-z", "-r",
        parent, authorization,
    ).stdout
    fields = [field for field in raw_diff.split(b"\0") if field]
    if len(fields) != 4:
        raise InfraRetryReleaseError("original authorization is not a two-path diff")
    observed = {
        (fields[index].decode("ascii"), fields[index + 1].decode("utf-8"))
        for index in (0, 2)
    }
    expected = {("M", ORIGINAL_PREREGISTRATION_PATH), ("A", ORIGINAL_MANIFEST_PATH)}
    if observed != expected:
        raise InfraRetryReleaseError("original authorization diff paths/statuses differ")
    _mode, manifest_blob = _tree_blob(repo, authorization, ORIGINAL_MANIFEST_PATH)
    if sha256_bytes(manifest_blob) != ORIGINAL_MANIFEST_SHA256:
        raise InfraRetryReleaseError("original manifest SHA-256 differs")
    manifest = _canonical_object(manifest_blob, "original Stage-T manifest")
    if manifest.get("static_root_commit") != parent:
        raise InfraRetryReleaseError("original manifest does not bind authorization parent")
    transition = manifest.get("status_transition")
    if not isinstance(transition, Mapping) or transition.get("path") != ORIGINAL_PREREGISTRATION_PATH:
        raise InfraRetryReleaseError("original manifest status transition differs")
    try:
        old = bytes.fromhex(transition["old_bytes_hex"])
        new = bytes.fromhex(transition["new_bytes_hex"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InfraRetryReleaseError("original manifest status bytes are malformed") from exc
    _mode, parent_prereg = _tree_blob(repo, parent, ORIGINAL_PREREGISTRATION_PATH)
    _mode, child_prereg = _tree_blob(repo, authorization, ORIGINAL_PREREGISTRATION_PATH)
    if parent_prereg.count(old) != 1 or child_prereg != parent_prereg.replace(old, new, 1):
        raise InfraRetryReleaseError("original authorization status change differs")
    return {
        "authorization_commit": authorization,
        "manifest_path": ORIGINAL_MANIFEST_PATH,
        "manifest_sha256": ORIGINAL_MANIFEST_SHA256,
        "static_root_commit": parent,
    }


def _fixed_retry_constraints() -> dict[str, Any]:
    return {
        "carry_in_conservative_spend_usd": CARRY_IN_CONSERVATIVE_SPEND_USD,
        "carry_in_observed_provider_delta_usd": CARRY_IN_OBSERVED_PROVIDER_DELTA_USD,
        "carry_in_provider_seconds": CARRY_IN_PROVIDER_SECONDS,
        "cumulative_provider_seconds_cap": CUMULATIVE_PROVIDER_SECONDS_CAP,
        "cumulative_spend_cap_usd": CUMULATIVE_SPEND_CAP_USD,
        "max_additional_provider_seconds": MAX_ADDITIONAL_PROVIDER_SECONDS,
        "max_additional_spend_usd": MAX_ADDITIONAL_SPEND_USD,
        "max_provider_allocations": 1,
        "max_authorization_invocations": 1,
        "consumption_record_relative_root": CONSUMPTION_ROOT_RELATIVE,
        "outer_authorization_is_sole_provider_authority": True,
        "fresh_inner_stage_t_receipts_permitted": 1,
        "inner_stage_t_receipt_provider_authority": False,
        "inner_stage_t_receipt_role": "bind-immutable-766db12-scientific-payload-only",
        "no_fallback": True,
        "pinned_cost_per_hour_usd": PINNED_COST_PER_HOUR_USD,
        "semantic_n": 0,
    }


def _verify_controller_root(repo: Path, root: str) -> str:
    controller = _exact_commit(
        repo, ONE_ATTEMPT_CONTROLLER_COMMIT, "one-attempt controller commit"
    )
    _require_ancestor(repo, controller, root, "one-attempt controller commit")
    for path in CONTROLLER_PATHS:
        root_mode, root_blob = _tree_blob(repo, root, path)
        controller_mode, controller_blob = _tree_blob(repo, controller, path)
        if (root_mode, root_blob) != (controller_mode, controller_blob):
            raise InfraRetryReleaseError(
                f"controller path differs from one-attempt controller commit: {path}"
            )
    return controller


def build_infra_retry_manifest(
    repo: Path, *, static_root_commit: str, manifest_path: str,
) -> dict[str, Any]:
    """Build a deterministic manifest from one immutable static-root tree."""
    repo = Path(repo).resolve()
    root = _exact_commit(repo, static_root_commit, "static root commit")
    manifest_path = _safe_path(manifest_path, "manifest path")
    if not _MANIFEST_PATH_RE.fullmatch(manifest_path):
        raise InfraRetryReleaseError("manifest path is outside retry-1 namespace")
    if not _path_absent(repo, root, manifest_path):
        raise InfraRetryReleaseError("retry manifest already exists in static root")
    patch = _exact_commit(repo, PROVIDER_PATCH_COMMIT, "provider patch commit")
    _require_ancestor(repo, patch, root, "provider patch commit")
    controller = _verify_controller_root(repo, root)
    original = _verify_original_authorization(repo, root)
    paths = _contract_paths(repo, root)
    _mode, amendment = _tree_blob(repo, root, AMENDMENT_PATH)
    if amendment.count(OLD_STATUS_LINE) != 1 or NEW_STATUS_LINE in amendment:
        raise InfraRetryReleaseError("static amendment is not uniquely DRAFT")
    _mode, incident = _tree_blob(repo, root, INCIDENT_PATH)
    if sha256_bytes(incident) != INCIDENT_SHA256:
        raise InfraRetryReleaseError("incident SHA-256 differs")
    inventory = _inventory(repo, root, paths)
    document = {
        "amendment_path": AMENDMENT_PATH,
        "design_id": DESIGN_ID,
        "failure_evidence_paths": list(FAILURE_EVIDENCE_PATHS),
        "inventory": inventory,
        "inventory_count": len(inventory),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(inventory)),
        "manifest_path": manifest_path,
        "original_scientific_payload": {**original, "semantic_n": 0},
        "provider_adapter": {
            "controller_paths": list(CONTROLLER_PATHS),
            "max_allocation_attempts": 1,
            "one_attempt_controller_commit": controller,
            "patch_commit": patch,
            "repair_scope": "runpod-real-allocation-response-schema-only",
        },
        "retry_constraints": _fixed_retry_constraints(),
        "schema": MANIFEST_SCHEMA,
        "stage": STAGE,
        "static_root_commit": root,
        "static_root_tree": _git_text(repo, "rev-parse", f"{root}^{{tree}}").strip(),
        "status_transition": {
            "new_bytes_hex": NEW_STATUS_LINE.hex(),
            "old_bytes_hex": OLD_STATUS_LINE.hex(),
            "path": AMENDMENT_PATH,
        },
    }
    verify_infra_retry_manifest(repo, document, expected_manifest_path=manifest_path)
    return document


def encode_infra_retry_manifest(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def verify_infra_retry_manifest(
    repo: Path, document: Mapping[str, Any], *, expected_manifest_path: str | None = None,
) -> dict[str, Any]:
    """Recompute every manifest binding from immutable Git objects."""
    if not isinstance(document, Mapping):
        raise InfraRetryReleaseError("retry manifest is not an object")
    expected_keys = {
        "amendment_path", "design_id", "failure_evidence_paths", "inventory",
        "inventory_count", "inventory_sha256", "manifest_path",
        "original_scientific_payload", "provider_adapter", "retry_constraints",
        "schema", "stage", "static_root_commit", "static_root_tree",
        "status_transition",
    }
    if set(document) != expected_keys:
        raise InfraRetryReleaseError("retry manifest fields differ")
    if document.get("schema") != MANIFEST_SCHEMA or document.get("design_id") != DESIGN_ID:
        raise InfraRetryReleaseError("retry manifest schema/design differs")
    if document.get("stage") != STAGE or document.get("amendment_path") != AMENDMENT_PATH:
        raise InfraRetryReleaseError("retry manifest stage/amendment differs")
    manifest_path = _safe_path(document.get("manifest_path"), "manifest path")
    if not _MANIFEST_PATH_RE.fullmatch(manifest_path):
        raise InfraRetryReleaseError("manifest path is outside retry-1 namespace")
    if expected_manifest_path is not None and manifest_path != expected_manifest_path:
        raise InfraRetryReleaseError("manifest path differs from expected path")
    root = _exact_commit(Path(repo), document.get("static_root_commit"), "static root commit")
    if document.get("static_root_tree") != _git_text(
        Path(repo), "rev-parse", f"{root}^{{tree}}"
    ).strip():
        raise InfraRetryReleaseError("static root tree differs")
    if not _path_absent(Path(repo), root, manifest_path):
        raise InfraRetryReleaseError("manifest path is not absent from static root")
    paths = _contract_paths(Path(repo), root)
    expected_inventory = _inventory(Path(repo), root, paths)
    if document.get("inventory") != expected_inventory:
        raise InfraRetryReleaseError("retry inventory differs from static root")
    if document.get("inventory_count") != len(expected_inventory):
        raise InfraRetryReleaseError("retry inventory count differs")
    if document.get("inventory_sha256") != sha256_bytes(canonical_json_bytes(expected_inventory)):
        raise InfraRetryReleaseError("retry inventory hash differs")
    if document.get("failure_evidence_paths") != list(FAILURE_EVIDENCE_PATHS):
        raise InfraRetryReleaseError("failure evidence path binding differs")
    original = _verify_original_authorization(Path(repo), root)
    if document.get("original_scientific_payload") != {**original, "semantic_n": 0}:
        raise InfraRetryReleaseError("original scientific payload binding differs")
    patch = _exact_commit(Path(repo), PROVIDER_PATCH_COMMIT, "provider patch commit")
    _require_ancestor(Path(repo), patch, root, "provider patch commit")
    controller = _verify_controller_root(Path(repo), root)
    expected_adapter = {
        "controller_paths": list(CONTROLLER_PATHS),
        "max_allocation_attempts": 1,
        "one_attempt_controller_commit": controller,
        "patch_commit": patch,
        "repair_scope": "runpod-real-allocation-response-schema-only",
    }
    if document.get("provider_adapter") != expected_adapter:
        raise InfraRetryReleaseError("provider adapter binding differs")
    if document.get("retry_constraints") != _fixed_retry_constraints():
        raise InfraRetryReleaseError("retry constraints differ")
    expected_transition = {
        "new_bytes_hex": NEW_STATUS_LINE.hex(),
        "old_bytes_hex": OLD_STATUS_LINE.hex(),
        "path": AMENDMENT_PATH,
    }
    if document.get("status_transition") != expected_transition:
        raise InfraRetryReleaseError("retry status transition differs")
    _mode, amendment = _tree_blob(Path(repo), root, AMENDMENT_PATH)
    if amendment.count(OLD_STATUS_LINE) != 1 or NEW_STATUS_LINE in amendment:
        raise InfraRetryReleaseError("static amendment is not uniquely DRAFT")
    return {"manifest_path": manifest_path, "static_root_commit": root, "status": "PASS"}


def verify_infra_retry_authorization(
    repo: Path, *, authorization_commit: str, manifest_path: str,
) -> dict[str, Any]:
    """Verify an exact single-parent, two-path authorization child."""
    repo = Path(repo).resolve()
    authorization = _exact_commit(repo, authorization_commit, "authorization commit")
    parents = _parents(repo, authorization)
    if len(parents) != 1:
        raise InfraRetryReleaseError("retry authorization must have exactly one parent")
    root = parents[0]
    raw = _git(
        repo, "diff-tree", "--no-commit-id", "--name-status", "-z", "-r",
        root, authorization,
    ).stdout
    fields = [field for field in raw.split(b"\0") if field]
    if len(fields) != 4:
        raise InfraRetryReleaseError("retry authorization is not an exact two-path diff")
    observed = {
        (fields[index].decode("ascii"), fields[index + 1].decode("utf-8"))
        for index in (0, 2)
    }
    expected = {("M", AMENDMENT_PATH), ("A", manifest_path)}
    if observed != expected:
        raise InfraRetryReleaseError("retry authorization diff paths/statuses differ")
    _mode, parent_amendment = _tree_blob(repo, root, AMENDMENT_PATH)
    _mode, child_amendment = _tree_blob(repo, authorization, AMENDMENT_PATH)
    if parent_amendment.count(OLD_STATUS_LINE) != 1 or child_amendment != parent_amendment.replace(
        OLD_STATUS_LINE, NEW_STATUS_LINE, 1
    ):
        raise InfraRetryReleaseError("amendment is not the exact status-only transition")
    _mode, manifest_blob = _tree_blob(repo, authorization, manifest_path)
    manifest = _canonical_object(manifest_blob, "retry manifest")
    verify_infra_retry_manifest(repo, manifest, expected_manifest_path=manifest_path)
    if manifest.get("static_root_commit") != root:
        raise InfraRetryReleaseError("retry manifest parent binding differs")
    return {
        "authorization_commit": authorization,
        "changed_paths": sorted((AMENDMENT_PATH, manifest_path)),
        "manifest": manifest,
        "manifest_sha256": sha256_bytes(manifest_blob),
        "static_root_commit": root,
        "status": "PASS",
    }


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise InfraRetryReleaseError("receipt timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _receipt_payload_sha256(document: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes({
        key: value for key, value in document.items() if key != "payload_sha256"
    }))


def create_infra_retry_receipt(
    repo: Path,
    receipt_directory: Path,
    *,
    authorization_commit: str,
    manifest_path: str,
    created_at: datetime | None = None,
) -> Path:
    """Create the only launch receipt in a new receipt directory."""
    evidence = verify_infra_retry_authorization(
        repo, authorization_commit=authorization_commit, manifest_path=manifest_path
    )
    directory = Path(receipt_directory)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise InfraRetryReleaseError("receipt directory already exists") from exc
    document = {
        "authorization_commit": evidence["authorization_commit"],
        "created_at": _utc_text(created_at or datetime.now(timezone.utc)),
        "manifest_path": manifest_path,
        "manifest_sha256": evidence["manifest_sha256"],
        "schema": RECEIPT_SCHEMA,
        "stage": STAGE,
        "static_root_commit": evidence["static_root_commit"],
    }
    document["payload_sha256"] = _receipt_payload_sha256(document)
    path = directory / RECEIPT_BASENAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(document) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise InfraRetryReleaseError("retry receipt already exists") from exc
    return path


def verify_infra_retry_checkout(
    repo: Path,
    *,
    authorization_commit: str,
    manifest_path: str,
    receipt_directory: Path,
    now: datetime | None = None,
    max_receipt_age_seconds: int = DEFAULT_MAX_RECEIPT_AGE_SECONDS,
    future_skew_seconds: int = DEFAULT_FUTURE_SKEW_SECONDS,
) -> dict[str, Any]:
    """Verify clean detached checkout, authorization, and one fresh receipt."""
    repo = Path(repo).resolve()
    evidence = verify_infra_retry_authorization(
        repo, authorization_commit=authorization_commit, manifest_path=manifest_path
    )
    head = _git_text(repo, "rev-parse", "HEAD").strip()
    if head != evidence["authorization_commit"]:
        raise InfraRetryReleaseError("checkout HEAD differs from retry authorization")
    symbolic = _git(repo, "symbolic-ref", "-q", "HEAD", check=False)
    if symbolic.returncode == 0:
        raise InfraRetryReleaseError("retry checkout is on a branch")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all").stdout:
        raise InfraRetryReleaseError("retry checkout is dirty")
    directory = Path(receipt_directory)
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise InfraRetryReleaseError(f"cannot inspect receipt directory: {exc}") from exc
    if len(entries) != 1 or entries[0].name != RECEIPT_BASENAME:
        raise InfraRetryReleaseError(f"expected one fixed retry receipt; observed {len(entries)}")
    path = entries[0]
    if path.is_symlink() or not path.is_file():
        raise InfraRetryReleaseError("retry receipt is not a regular non-symlink file")
    receipt = _canonical_object(path.read_bytes(), "retry receipt")
    expected_keys = {
        "authorization_commit", "created_at", "manifest_path", "manifest_sha256",
        "payload_sha256", "schema", "stage", "static_root_commit",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != RECEIPT_SCHEMA:
        raise InfraRetryReleaseError("retry receipt schema/fields differ")
    if receipt.get("stage") != STAGE or receipt.get("authorization_commit") != authorization_commit:
        raise InfraRetryReleaseError("retry receipt stage/authorization differs")
    if receipt.get("manifest_path") != manifest_path or receipt.get("manifest_sha256") != evidence["manifest_sha256"]:
        raise InfraRetryReleaseError("retry receipt manifest binding differs")
    if receipt.get("static_root_commit") != evidence["static_root_commit"]:
        raise InfraRetryReleaseError("retry receipt static-root binding differs")
    if receipt.get("payload_sha256") != _receipt_payload_sha256(receipt):
        raise InfraRetryReleaseError("retry receipt payload hash differs")
    try:
        created = datetime.strptime(receipt["created_at"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=timezone.utc
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InfraRetryReleaseError("retry receipt timestamp is malformed") from exc
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (current - created).total_seconds()
    if age > max_receipt_age_seconds:
        raise InfraRetryReleaseError("retry receipt is stale")
    if age < -future_skew_seconds:
        raise InfraRetryReleaseError("retry receipt is from the future")
    return {
        "authorization": evidence,
        "detached_head": head,
        "receipt": {**receipt, "age_seconds": age},
        "status": "PASS",
    }


__all__ = [
    "AMENDMENT_PATH", "CONTRACT_PATH", "InfraRetryReleaseError",
    "MANIFEST_SCHEMA", "NEW_STATUS_LINE", "OLD_STATUS_LINE", "RECEIPT_BASENAME",
    "build_infra_retry_manifest", "canonical_json_bytes", "create_infra_retry_receipt",
    "encode_infra_retry_manifest", "sha256_bytes", "verify_infra_retry_authorization",
    "verify_infra_retry_checkout", "verify_infra_retry_manifest",
]
