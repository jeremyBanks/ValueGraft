"""Append-only persistence and release bindings for decision-canary v12.

This module is deliberately store-only.  It imports no earlier experiment
schema and executes no model code.  A checkpoint is immutable once created;
promotion writes a new checkpoint that binds the literal bytes of its parent.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping, Sequence


DESIGN_ID = "coherent-state-decision-canary-v12"
SCHEMA_ID = "coherent-state-decision-canary-v12-store-v1"
MODES = ("technical", "eligibility", "treatment")
REGION_IDS = ("R1_content", "R2_boundary", "R3_anchor")
HISTORY_IDS = ("F", "C", "W")

MODE_STAGES = {
    "technical": ("STARTED", "CAPTURED", "VALIDATED", ("PASS", "FAIL")),
    "eligibility": (
        "STARTED", "MECHANICAL_PASS", "REVIEW_PASS", ("PASS", "FAIL")
    ),
    "treatment": ("RELEASED", "CAPTURED", "SCORED", ("COMPLETE", "INVALID")),
}
INITIAL_STAGE = {
    "technical": "STARTED",
    "eligibility": "STARTED",
    "treatment": "RELEASED",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CanaryStoreError(RuntimeError):
    """A checkpoint, bound artifact, or release condition failed closed."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise CanaryStoreError("UTC timestamp input is timezone-naive")
    current = current.astimezone(timezone.utc)
    # Microseconds prevent two append-only promotions in one second colliding.
    return current.strftime("%Y%m%dT%H%M%S%fZ")


def unique_output_path(
    output_dir: Path, experiment: str, model_slug: str, *, timestamp: str | None = None
) -> Path:
    """Return the required self-announcing path, refusing an existing name."""
    if not SLUG_RE.fullmatch(experiment) or not SLUG_RE.fullmatch(model_slug):
        raise CanaryStoreError("experiment/model slug is empty or unsafe")
    stamp = timestamp or utc_timestamp()
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}(?:[0-9]{6})?Z", stamp):
        raise CanaryStoreError("output timestamp is not a UTC filename timestamp")
    path = output_dir / f"{experiment}_{model_slug}_{stamp}.json"
    if path.exists():
        raise CanaryStoreError(f"refusing non-unique output path: {path}")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_create_json(path: Path, value: Any) -> None:
    """Atomically create JSON without ever replacing an existing checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # link is an atomic create-if-absent operation; unlike replace it
            # cannot erase an earlier checkpoint after a race.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CanaryStoreError(f"checkpoint already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except Exception as exc:
        raise CanaryStoreError(f"cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise CanaryStoreError("checkpoint root is not an object")
    validate_checkpoint(document)
    return document


def _stage_rank(mode: str, stage: str) -> tuple[int, bool]:
    if mode not in MODES:
        raise CanaryStoreError(f"unknown checkpoint mode: {mode}")
    stages = MODE_STAGES[mode]
    for rank, entry in enumerate(stages):
        if isinstance(entry, tuple):
            if stage in entry:
                return rank, True
        elif stage == entry:
            return rank, False
    raise CanaryStoreError(f"unknown {mode} stage: {stage}")


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise CanaryStoreError(f"{label} is not a lowercase SHA-256")
    return value


def _normalise_bindings(
    bindings: Sequence[Mapping[str, Any]], label: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in bindings:
        if set(item) != {"path", "sha256"}:
            raise CanaryStoreError(f"{label} binding fields differ")
        path = item["path"]
        if not isinstance(path, str) or not path or path in seen:
            raise CanaryStoreError(f"{label} binding path is empty or repeated")
        _validate_hash(item["sha256"], f"{label} binding hash")
        seen.add(path)
        rows.append({"path": path, "sha256": item["sha256"]})
    return sorted(rows, key=lambda row: row["path"])


def literal_file_binding(path: Path, *, display_path: str | None = None) -> dict[str, str]:
    if not path.is_file():
        raise CanaryStoreError(f"bound file is absent: {path}")
    return {"path": display_path or str(path), "sha256": file_sha256(path)}


def validate_bounds(bounds: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the explicit one-case/three-history/three-region row bounds."""
    expected = {
        "case_count", "history_ids", "region_ids", "layer_indices",
        "token_counts_by_region", "limits",
    }
    if set(bounds) != expected:
        raise CanaryStoreError("state-bound field set differs")
    limits = bounds["limits"]
    if not isinstance(limits, Mapping) or set(limits) != {
        "max_cases", "max_histories", "max_regions", "max_layers",
        "max_tokens_per_region",
    }:
        raise CanaryStoreError("state-bound limit field set differs")
    if limits["max_cases"] != 1 or limits["max_histories"] != 3 or \
            limits["max_regions"] != 3:
        raise CanaryStoreError("store must assert one case/three histories/three regions")
    if not isinstance(limits["max_layers"], int) or limits["max_layers"] < 1:
        raise CanaryStoreError("max_layers is not an explicit positive integer")
    if not isinstance(limits["max_tokens_per_region"], int) or \
            limits["max_tokens_per_region"] < 1:
        raise CanaryStoreError("max_tokens_per_region is not explicit and positive")
    if bounds["case_count"] != 1:
        raise CanaryStoreError("a checkpoint must contain exactly one case")

    histories = list(bounds["history_ids"])
    if not (1 <= len(histories) <= 3) or len(histories) != len(set(histories)) or \
            any(item not in HISTORY_IDS for item in histories):
        raise CanaryStoreError("history bound exceeds or differs from F/C/W")
    regions = list(bounds["region_ids"])
    if not (1 <= len(regions) <= 3) or len(regions) != len(set(regions)) or \
            any(item not in REGION_IDS for item in regions):
        raise CanaryStoreError("region bound exceeds or differs from R1/R2/R3")
    layers = list(bounds["layer_indices"])
    if not (1 <= len(layers) <= limits["max_layers"]) or \
            len(layers) != len(set(layers)) or \
            any(not isinstance(item, int) or item < 0 for item in layers):
        raise CanaryStoreError("layer indices exceed the asserted layer bound")
    token_counts = bounds["token_counts_by_region"]
    if not isinstance(token_counts, Mapping) or set(token_counts) != set(regions):
        raise CanaryStoreError("token-count regions differ from retained regions")
    for region, count in token_counts.items():
        if not isinstance(count, int) or not 1 <= count <= limits["max_tokens_per_region"]:
            raise CanaryStoreError(f"token count exceeds the asserted bound: {region}")
    return deepcopy(dict(bounds))


def inventory_external_tensor(
    path: Path,
    *,
    display_path: str,
    case_id: str,
    history_ids: Sequence[str],
    region_ids: Sequence[str],
    layer_indices: Sequence[int],
    token_counts_by_region: Mapping[str, int],
    dtype: str,
    shape: Sequence[int],
    retention: str,
) -> dict[str, Any]:
    """Build a literal inventory row for a lossless external tensor artifact."""
    if not path.is_file():
        raise CanaryStoreError(f"external tensor is absent: {path}")
    if not dtype or not retention or not case_id or not display_path:
        raise CanaryStoreError("external tensor metadata is incomplete")
    if not shape or any(not isinstance(value, int) or value < 1 for value in shape):
        raise CanaryStoreError("external tensor shape is invalid")
    return {
        "kind": "tensor",
        "path": display_path,
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "case_id": case_id,
        "history_ids": list(history_ids),
        "region_ids": list(region_ids),
        "layer_indices": list(layer_indices),
        "token_counts_by_region": dict(token_counts_by_region),
        "dtype": dtype,
        "shape": list(shape),
        "retention": retention,
    }


def _validate_artifacts(
    artifacts: Sequence[Mapping[str, Any]], bounds: Mapping[str, Any], case_id: str
) -> list[dict[str, Any]]:
    expected = {
        "kind", "path", "sha256", "size_bytes", "case_id", "history_ids",
        "region_ids", "layer_indices", "token_counts_by_region", "dtype",
        "shape", "retention",
    }
    result: list[dict[str, Any]] = []
    paths: set[str] = set()
    for row in artifacts:
        if set(row) != expected or row.get("kind") != "tensor":
            raise CanaryStoreError("external artifact inventory fields/kind differ")
        if row["case_id"] != case_id:
            raise CanaryStoreError("external tensor case binding differs")
        if not isinstance(row["path"], str) or not row["path"] or row["path"] in paths:
            raise CanaryStoreError("external tensor path is empty or repeated")
        paths.add(row["path"])
        _validate_hash(row["sha256"], "external tensor hash")
        if not isinstance(row["size_bytes"], int) or row["size_bytes"] < 1:
            raise CanaryStoreError("external tensor size is invalid")
        if not set(row["history_ids"]).issubset(bounds["history_ids"]):
            raise CanaryStoreError("external tensor histories exceed state bounds")
        if not set(row["region_ids"]).issubset(bounds["region_ids"]):
            raise CanaryStoreError("external tensor regions exceed state bounds")
        if not set(row["layer_indices"]).issubset(bounds["layer_indices"]):
            raise CanaryStoreError("external tensor layers exceed state bounds")
        if set(row["token_counts_by_region"]) != set(row["region_ids"]):
            raise CanaryStoreError("external tensor token regions differ")
        for region, count in row["token_counts_by_region"].items():
            if count > bounds["token_counts_by_region"][region] or count < 1:
                raise CanaryStoreError("external tensor tokens exceed state bounds")
        if not isinstance(row["dtype"], str) or not row["dtype"] or \
                not isinstance(row["retention"], str) or not row["retention"]:
            raise CanaryStoreError("external tensor dtype/retention is incomplete")
        if not isinstance(row["shape"], list) or not row["shape"] or any(
                not isinstance(value, int) or value < 1 for value in row["shape"]):
            raise CanaryStoreError("external tensor shape is invalid")
        result.append(deepcopy(dict(row)))
    return sorted(result, key=lambda row: row["path"])


def _status_for(mode: str, stage: str) -> str:
    _, terminal = _stage_rank(mode, stage)
    if mode in ("technical", "eligibility"):
        return stage if terminal else "IN_PROGRESS"
    if stage == "COMPLETE":
        return "PASS"
    if stage == "INVALID":
        return "FAIL"
    return "IN_PROGRESS"


def _base_document(
    *,
    mode: str,
    stage: str,
    sequence: int,
    case_id: str,
    fingerprint: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    review_bindings: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    artifact_inventory: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    previous_checkpoint: Mapping[str, str] | None,
    release_receipts: Mapping[str, Mapping[str, str]] | None,
    created_at_utc: str,
) -> dict[str, Any]:
    rank, _ = _stage_rank(mode, stage)
    if not isinstance(case_id, str) or not case_id:
        raise CanaryStoreError("case_id is empty")
    if not isinstance(fingerprint, Mapping) or not fingerprint:
        raise CanaryStoreError("environment/repository fingerprint is empty")
    sources = _normalise_bindings(source_bindings, "source")
    reviews = _normalise_bindings(review_bindings, "review")
    if not sources:
        raise CanaryStoreError("source bindings are empty")
    if mode in ("eligibility", "treatment") and not reviews:
        raise CanaryStoreError(f"{mode} review bindings are empty")
    frozen_bounds = validate_bounds(bounds)
    inventory = _validate_artifacts(artifact_inventory, frozen_bounds, case_id)
    if not isinstance(payload, Mapping):
        raise CanaryStoreError("checkpoint payload is not an object")
    return {
        "schema": SCHEMA_ID,
        "design_id": DESIGN_ID,
        "mode": mode,
        "stage": stage,
        "stage_rank": rank,
        "status": _status_for(mode, stage),
        "sequence": sequence,
        "case_id": case_id,
        "created_at_utc": created_at_utc,
        "fingerprint": deepcopy(dict(fingerprint)),
        "fingerprint_sha256": canonical_sha256(fingerprint),
        "source_bindings": sources,
        "source_bindings_sha256": canonical_sha256(sources),
        "review_bindings": reviews,
        "review_bindings_sha256": canonical_sha256(reviews),
        "bounds": frozen_bounds,
        "artifact_inventory": inventory,
        "payload": deepcopy(dict(payload)),
        "previous_checkpoint": deepcopy(previous_checkpoint),
        "release_receipts": deepcopy(release_receipts),
    }


def validate_checkpoint(document: Mapping[str, Any]) -> None:
    expected = {
        "schema", "design_id", "mode", "stage", "stage_rank", "status",
        "sequence", "case_id", "created_at_utc", "fingerprint",
        "fingerprint_sha256", "source_bindings", "source_bindings_sha256",
        "review_bindings", "review_bindings_sha256", "bounds",
        "artifact_inventory", "payload", "previous_checkpoint",
        "release_receipts",
    }
    if set(document) != expected:
        raise CanaryStoreError("checkpoint field set differs")
    if document["schema"] != SCHEMA_ID or document["design_id"] != DESIGN_ID:
        raise CanaryStoreError("checkpoint schema/design differs")
    rank, _ = _stage_rank(document["mode"], document["stage"])
    if document["stage_rank"] != rank or \
            document["status"] != _status_for(document["mode"], document["stage"]):
        raise CanaryStoreError("checkpoint stage/status differs")
    if not isinstance(document["sequence"], int) or document["sequence"] < 0:
        raise CanaryStoreError("checkpoint sequence is invalid")
    if not isinstance(document["case_id"], str) or not document["case_id"]:
        raise CanaryStoreError("checkpoint case is invalid")
    if not isinstance(document["created_at_utc"], str) or not re.fullmatch(
            r"[0-9]{8}T[0-9]{6}(?:[0-9]{6})?Z", document["created_at_utc"]):
        raise CanaryStoreError("checkpoint creation timestamp is invalid")
    if not isinstance(document["fingerprint"], Mapping) or \
            document["fingerprint_sha256"] != canonical_sha256(document["fingerprint"]):
        raise CanaryStoreError("checkpoint fingerprint binding differs")
    sources = _normalise_bindings(document["source_bindings"], "source")
    reviews = _normalise_bindings(document["review_bindings"], "review")
    if sources != document["source_bindings"] or \
            document["source_bindings_sha256"] != canonical_sha256(sources):
        raise CanaryStoreError("checkpoint source binding differs")
    if reviews != document["review_bindings"] or \
            document["review_bindings_sha256"] != canonical_sha256(reviews):
        raise CanaryStoreError("checkpoint review binding differs")
    if not sources or document["mode"] in ("eligibility", "treatment") and not reviews:
        raise CanaryStoreError("checkpoint omits required source/review bindings")
    bounds = validate_bounds(document["bounds"])
    artifacts = _validate_artifacts(
        document["artifact_inventory"], bounds, document["case_id"]
    )
    if artifacts != document["artifact_inventory"]:
        raise CanaryStoreError("checkpoint artifact inventory order differs")
    if not isinstance(document["payload"], Mapping):
        raise CanaryStoreError("checkpoint payload is invalid")
    previous = document["previous_checkpoint"]
    if document["sequence"] == 0:
        if previous is not None:
            raise CanaryStoreError("initial checkpoint has a parent")
    elif not isinstance(previous, Mapping) or set(previous) != {"path", "sha256"}:
        raise CanaryStoreError("promoted checkpoint lacks exact parent binding")
    elif not isinstance(previous["path"], str) or not previous["path"]:
        raise CanaryStoreError("promoted checkpoint parent path is empty")
    else:
        _validate_hash(previous["sha256"], "parent checkpoint hash")
    if document["mode"] == "treatment":
        receipts = document["release_receipts"]
        if not isinstance(receipts, Mapping) or set(receipts) != {
            "technical", "eligibility"
        }:
            raise CanaryStoreError("treatment checkpoint lacks both release receipts")
        for mode, binding in receipts.items():
            if not isinstance(binding, Mapping) or set(binding) != {
                "path", "sha256", "status", "design_id"
            }:
                raise CanaryStoreError(f"{mode} release-receipt binding differs")
            _validate_hash(binding["sha256"], f"{mode} release-receipt hash")
            if binding["status"] != "PASS" or binding["design_id"] != DESIGN_ID:
                raise CanaryStoreError(f"{mode} release-receipt status/design differs")
    elif document["release_receipts"] is not None:
        raise CanaryStoreError("non-treatment checkpoint contains release receipts")


def create_checkpoint(
    path: Path,
    *,
    mode: str,
    case_id: str,
    fingerprint: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    review_bindings: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    artifact_inventory: Sequence[Mapping[str, Any]] = (),
    payload: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create the immutable first technical or eligibility checkpoint."""
    if mode not in MODES:
        raise CanaryStoreError(f"unknown checkpoint mode: {mode}")
    if mode == "treatment":
        raise CanaryStoreError("treatment creation requires committed release receipts")
    document = _base_document(
        mode=mode,
        stage=INITIAL_STAGE[mode],
        sequence=0,
        case_id=case_id,
        fingerprint=fingerprint,
        source_bindings=source_bindings,
        review_bindings=review_bindings,
        bounds=bounds,
        artifact_inventory=artifact_inventory,
        payload=payload or {},
        previous_checkpoint=None,
        release_receipts=None,
        created_at_utc=created_at_utc or utc_timestamp(),
    )
    validate_checkpoint(document)
    atomic_create_json(path, document)
    return document


def _deep_extend(old: Mapping[str, Any], additions: Mapping[str, Any], prefix: str = "") -> dict:
    result = deepcopy(dict(old))
    for key, value in additions.items():
        location = f"{prefix}.{key}" if prefix else key
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_extend(result[key], value, location)
        elif result[key] != value:
            raise CanaryStoreError(f"promotion would overwrite payload field: {location}")
    return result


def promote_checkpoint(
    prior_path: Path,
    new_path: Path,
    *,
    stage: str,
    payload_additions: Mapping[str, Any] | None = None,
    artifact_additions: Sequence[Mapping[str, Any]] = (),
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Promote by creating a child; neither the parent nor old fields change."""
    prior = read_checkpoint(prior_path)
    old_rank, old_terminal = _stage_rank(prior["mode"], prior["stage"])
    new_rank, _ = _stage_rank(prior["mode"], stage)
    if old_terminal or new_rank != old_rank + 1:
        raise CanaryStoreError(
            f"non-monotonic/terminal stage promotion {prior['stage']} -> {stage}"
        )
    payload = _deep_extend(prior["payload"], payload_additions or {})
    inventory_by_path = {row["path"]: row for row in prior["artifact_inventory"]}
    for row in artifact_additions:
        path = row.get("path") if isinstance(row, Mapping) else None
        if path in inventory_by_path and inventory_by_path[path] != row:
            raise CanaryStoreError(f"promotion would overwrite artifact inventory: {path}")
        if path in inventory_by_path:
            continue
        inventory_by_path[path] = deepcopy(dict(row))
    document = _base_document(
        mode=prior["mode"],
        stage=stage,
        sequence=prior["sequence"] + 1,
        case_id=prior["case_id"],
        fingerprint=prior["fingerprint"],
        source_bindings=prior["source_bindings"],
        review_bindings=prior["review_bindings"],
        bounds=prior["bounds"],
        artifact_inventory=list(inventory_by_path.values()),
        payload=payload,
        previous_checkpoint={"path": str(prior_path), "sha256": file_sha256(prior_path)},
        release_receipts=prior["release_receipts"],
        created_at_utc=created_at_utc or utc_timestamp(),
    )
    validate_checkpoint(document)
    atomic_create_json(new_path, document)
    return document


def _safe_repo_relative(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts or pure.parts[0] == ".git":
        raise CanaryStoreError(f"unsafe receipt path: {path}")
    return pure.as_posix()


def _git(repo: Path, *args: str, text: bool = False) -> bytes | str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=text,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr if text else exc.stderr.decode("utf-8", "replace")
        raise CanaryStoreError(f"git receipt verification failed: {detail.strip()}") from exc
    return result.stdout


def verify_head_receipt(
    repo_root: Path, expectation: Mapping[str, str], *, expected_mode: str
) -> tuple[dict[str, Any], dict[str, str]]:
    """Verify expected literal bytes exist both in HEAD and the worktree."""
    if set(expectation) != {"path", "sha256", "status", "design_id"}:
        raise CanaryStoreError(f"{expected_mode} receipt expectation fields differ")
    relative = _safe_repo_relative(expectation["path"])
    _validate_hash(expectation["sha256"], f"{expected_mode} expected receipt hash")
    if expectation["status"] != "PASS" or expectation["design_id"] != DESIGN_ID:
        raise CanaryStoreError(f"{expected_mode} expected receipt status/design differs")
    raw = _git(repo_root, "show", f"HEAD:{relative}")
    if not isinstance(raw, bytes):
        raise AssertionError("binary git show unexpectedly returned text")
    observed_hash = sha256_bytes(raw)
    if observed_hash != expectation["sha256"]:
        raise CanaryStoreError(f"{expected_mode} literal HEAD receipt SHA-256 differs")
    worktree_path = repo_root / relative
    if not worktree_path.is_file() or worktree_path.read_bytes() != raw:
        raise CanaryStoreError(f"{expected_mode} receipt is not identical to tracked HEAD")
    try:
        receipt = json.loads(raw)
    except Exception as exc:
        raise CanaryStoreError(f"{expected_mode} HEAD receipt is not JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise CanaryStoreError(f"{expected_mode} HEAD receipt root is not an object")
    validate_checkpoint(receipt)
    if receipt["mode"] != expected_mode or receipt["status"] != expectation["status"] or \
            receipt["design_id"] != expectation["design_id"]:
        raise CanaryStoreError(f"{expected_mode} receipt mode/status/design differs")
    return receipt, dict(expectation)


def create_treatment_checkpoint(
    path: Path,
    *,
    repo_root: Path,
    case_id: str,
    fingerprint: Mapping[str, Any],
    source_bindings: Sequence[Mapping[str, Any]],
    review_bindings: Sequence[Mapping[str, Any]],
    bounds: Mapping[str, Any],
    technical_receipt: Mapping[str, str],
    eligibility_receipt: Mapping[str, str],
    artifact_inventory: Sequence[Mapping[str, Any]] = (),
    payload: Mapping[str, Any] | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Release treatment only from exact PASS receipts committed in HEAD."""
    technical, technical_binding = verify_head_receipt(
        repo_root, technical_receipt, expected_mode="technical"
    )
    eligibility, eligibility_binding = verify_head_receipt(
        repo_root, eligibility_receipt, expected_mode="eligibility"
    )
    source_rows = _normalise_bindings(source_bindings, "source")
    review_rows = _normalise_bindings(review_bindings, "review")
    if technical["fingerprint_sha256"] != canonical_sha256(fingerprint):
        raise CanaryStoreError("technical receipt fingerprint does not bind treatment")
    if eligibility["source_bindings_sha256"] != canonical_sha256(source_rows):
        raise CanaryStoreError("eligibility receipt sources do not bind treatment")
    if eligibility["review_bindings_sha256"] != canonical_sha256(review_rows):
        raise CanaryStoreError("eligibility receipt reviews do not bind treatment")
    # The exact-stack technical gate is a reusable literal fixture, not an eNN
    # semantic case.  Eligibility, in contrast, is deliberately case-specific.
    if eligibility["case_id"] != case_id:
        raise CanaryStoreError("eligibility receipt case does not bind treatment case")
    document = _base_document(
        mode="treatment",
        stage="RELEASED",
        sequence=0,
        case_id=case_id,
        fingerprint=fingerprint,
        source_bindings=source_rows,
        review_bindings=review_rows,
        bounds=bounds,
        artifact_inventory=artifact_inventory,
        payload=payload or {},
        previous_checkpoint=None,
        release_receipts={
            "technical": technical_binding,
            "eligibility": eligibility_binding,
        },
        created_at_utc=created_at_utc or utc_timestamp(),
    )
    validate_checkpoint(document)
    atomic_create_json(path, document)
    return document
