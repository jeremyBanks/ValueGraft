"""Bounded lossless selected-state bundles for powered successor v13.

One bundle contains a fresh cache boundary through dynamic R2 and exact C/W
selected R2 rows for one case/render/schedule.  Files load on CPU for audit.
Treatment must call :func:`materialize_verified_bundle` explicitly before any
continuation.  The module contains no model forward or outcome logic.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from coherent_canary_runtime import Snapshot, tensor_sha256
from powered_v13_schema import DESIGN_ID, N_SCHEDULE, P_SCHEDULE, PRIMARY_ARMS


SCHEMA = "coherent-state-powered-successor-v13-r2-bundle-v1"
HISTORIES = ("F", "C", "W")
SOURCE_HISTORIES = ("C", "W")
SCHEDULES = (N_SCHEDULE, P_SCHEDULE)
MAX_LAYERS = 48
REQUIRED_LAYERS = 48
MAX_KV_HEADS = 8
MAX_HEAD_DIM = 256
MAX_FRESH_BOUNDARY_TOKENS = 1024
MAX_SELECTED_ROWS = 256
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class V13BundleError(RuntimeError):
    """A bundle or executable materialization violates its exact contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13BundleError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13BundleError(f"value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} is not lowercase SHA-256")
    return value


def _slug(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SLUG.fullmatch(value) is not None,
             f"{label} is not a safe nonempty slug")
    return value


def _plain_int(value: object, label: str, *, minimum: int,
               maximum: int) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool)
             and minimum <= value <= maximum,
             f"{label} lies outside {minimum}..{maximum}")
    return value


def _state_key(descriptor: Mapping[str, Any], history: str) -> list[str]:
    return [
        DESIGN_ID,
        descriptor["release_sha256"],
        descriptor["runtime_fingerprint_sha256"],
        descriptor["case_id"],
        descriptor["render_id"],
        descriptor["schedule"],
        history,
    ]


def validate_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "design_id", "release_sha256",
        "runtime_fingerprint_sha256", "primary_batch_id", "gpu_uuid",
        "case_id", "render_id", "schedule", "history_ids", "state_keys",
        "plan_sha256s", "foundation_bindings", "selected_map_sha256",
        "selected_token_ids_sha256", "fresh_boundary_token_count",
        "destination", "source", "cache_geometry",
    }
    _require(isinstance(value, Mapping) and set(value) == expected,
             "bundle descriptor field set differs")
    _require(value.get("schema") == SCHEMA and value.get("design_id") == DESIGN_ID,
             "bundle descriptor schema/design differs")
    _sha(value.get("release_sha256"), "release_sha256")
    _sha(value.get("runtime_fingerprint_sha256"), "runtime_fingerprint_sha256")
    for key in ("primary_batch_id", "case_id"):
        _slug(value.get(key), key)
    _require(value.get("render_id") in ("r1", "r2"),
             "render_id must be r1 or r2")
    _require(isinstance(value.get("gpu_uuid"), str)
             and bool(value["gpu_uuid"].strip()), "gpu_uuid is empty")
    _require(value.get("schedule") in SCHEDULES, "bundle schedule differs")
    _require(value.get("history_ids") == list(HISTORIES),
             "bundle history order differs")

    plans = value.get("plan_sha256s")
    _require(isinstance(plans, Mapping) and set(plans) == set(HISTORIES),
             "bundle plan bindings differ")
    for history in HISTORIES:
        _sha(plans[history], f"{history} plan")
    bindings = value.get("foundation_bindings")
    _require(isinstance(bindings, Mapping) and set(bindings) == {
        "fixture_sha256", "render_attempt_sha256", "review_sha256",
        "phase_a_sha256",
    }, "foundation binding fields differ")
    for name, digest in bindings.items():
        _sha(digest, f"foundation {name}")
    _sha(value.get("selected_map_sha256"), "selected_map_sha256")
    _sha(value.get("selected_token_ids_sha256"),
         "selected_token_ids_sha256")

    geometry = value.get("cache_geometry")
    _require(isinstance(geometry, Mapping) and set(geometry) == {
        "layers", "batch", "kv_heads", "head_dim", "dtype",
    }, "bundle cache geometry fields differ")
    _require(geometry.get("layers") == REQUIRED_LAYERS
             and geometry.get("batch") == 1
             and isinstance(geometry.get("kv_heads"), int)
             and not isinstance(geometry.get("kv_heads"), bool)
             and 1 <= geometry["kv_heads"] <= MAX_KV_HEADS
             and isinstance(geometry.get("head_dim"), int)
             and not isinstance(geometry.get("head_dim"), bool)
             and 1 <= geometry["head_dim"] <= MAX_HEAD_DIM
             and geometry.get("dtype") == "torch.bfloat16",
             "bundle cache geometry exceeds frozen bounds")
    fresh_tokens = _plain_int(
        value.get("fresh_boundary_token_count"),
        "fresh_boundary_token_count", minimum=1,
        maximum=MAX_FRESH_BOUNDARY_TOKENS)

    destination = value.get("destination")
    _require(isinstance(destination, Mapping) and set(destination) == {
        "r1_start", "r1_end", "r2_end",
    }, "bundle destination fields differ")
    points = [destination.get(name) for name in ("r1_start", "r1_end", "r2_end")]
    _require(all(isinstance(point, int) and not isinstance(point, bool)
                 for point in points), "bundle destination points are not integers")
    start, content_end, end = map(int, points)
    _require(0 <= start < content_end < end == fresh_tokens,
             "bundle destination R1/R2 geometry differs")
    _require(end - start <= MAX_SELECTED_ROWS,
             "bundle selected R2 rows exceed frozen bound")

    source = value.get("source")
    _require(isinstance(source, Mapping) and set(source) == set(SOURCE_HISTORIES),
             "bundle source history fields differ")
    for history in SOURCE_HISTORIES:
        row = source[history]
        _require(isinstance(row, Mapping) and set(row) == {
            "logical_start", "logical_end", "selected_rows_sha256",
        }, f"bundle {history} source fields differ")
        logical_start = row.get("logical_start")
        logical_end = row.get("logical_end")
        _require(isinstance(logical_start, int) and not isinstance(logical_start, bool)
                 and isinstance(logical_end, int) and not isinstance(logical_end, bool)
                 and 0 <= logical_start < logical_end
                 and logical_end - logical_start == end - start,
                 f"bundle {history} source geometry differs")
        _sha(row.get("selected_rows_sha256"),
             f"bundle {history} selected rows")

    state_keys = value.get("state_keys")
    _require(isinstance(state_keys, Mapping) and set(state_keys) == set(HISTORIES),
             "bundle state-key histories differ")
    for history in HISTORIES:
        _require(state_keys[history] == _state_key(value, history),
                 f"bundle {history} state key differs")
    return deepcopy(dict(value))


def build_descriptor(
    *,
    release_sha256: str,
    runtime_fingerprint_sha256: str,
    primary_batch_id: str,
    gpu_uuid: str,
    case_id: str,
    render_id: str,
    schedule: str,
    plan_sha256s: Mapping[str, str],
    foundation_bindings: Mapping[str, str],
    selected_map_sha256: str,
    selected_token_ids_sha256: str,
    fresh_boundary_token_count: int,
    r1_start: int,
    r1_end: int,
    r2_end: int,
    source_intervals: Mapping[str, tuple[int, int]],
    source_rows_sha256: Mapping[str, str],
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "release_sha256": release_sha256,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "primary_batch_id": primary_batch_id,
        "gpu_uuid": gpu_uuid,
        "case_id": case_id,
        "render_id": render_id,
        "schedule": schedule,
        "history_ids": list(HISTORIES),
        "state_keys": {},
        "plan_sha256s": dict(plan_sha256s),
        "foundation_bindings": dict(foundation_bindings),
        "selected_map_sha256": selected_map_sha256,
        "selected_token_ids_sha256": selected_token_ids_sha256,
        "fresh_boundary_token_count": fresh_boundary_token_count,
        "destination": {
            "r1_start": r1_start, "r1_end": r1_end, "r2_end": r2_end,
        },
        "source": {
            history: {
                "logical_start": source_intervals[history][0],
                "logical_end": source_intervals[history][1],
                "selected_rows_sha256": source_rows_sha256[history],
            }
            for history in SOURCE_HISTORIES
        },
        "cache_geometry": {
            "layers": REQUIRED_LAYERS,
            "batch": 1,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "dtype": "torch.bfloat16",
        },
    }
    descriptor["state_keys"] = {
        history: _state_key(descriptor, history) for history in HISTORIES}
    return validate_descriptor(descriptor)


def _snapshot_geometry(snapshot: Snapshot, label: str,
                       *, max_tokens: int) -> dict[str, int | str]:
    _require(isinstance(snapshot, list) and len(snapshot) == REQUIRED_LAYERS,
             f"{label} must contain exactly {REQUIRED_LAYERS} layers")
    reference: dict[str, int | str] | None = None
    for layer, pair in enumerate(snapshot):
        _require(isinstance(pair, tuple) and len(pair) == 2,
                 f"{label} layer {layer} is not a K/V pair")
        keys, values = pair
        _require(isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor)
                 and keys.shape == values.shape and keys.ndim == values.ndim == 4,
                 f"{label} layer {layer} K/V geometry differs")
        _require(keys.device.type == values.device.type == "cpu",
                 f"{label} layer {layer} is not evicted to CPU")
        _require(keys.dtype == values.dtype == torch.bfloat16,
                 f"{label} layer {layer} is not exact bf16")
        _require(torch.isfinite(keys).all().item()
                 and torch.isfinite(values).all().item(),
                 f"{label} layer {layer} contains nonfinite state")
        batch, heads, tokens, head_dim = map(int, keys.shape)
        _require(batch == 1 and 1 <= heads <= MAX_KV_HEADS
                 and 1 <= tokens <= max_tokens and 1 <= head_dim <= MAX_HEAD_DIM,
                 f"{label} layer {layer} exceeds frozen geometry")
        observed: dict[str, int | str] = {
            "batch": batch, "kv_heads": heads, "tokens": tokens,
            "head_dim": head_dim, "dtype": str(keys.dtype),
        }
        if reference is None:
            reference = observed
        _require(observed == reference, f"{label} layer geometry varies")
    assert reference is not None
    return {"layers": len(snapshot), **reference}


def snapshot_sha256(snapshot: Snapshot) -> str:
    rows = [
        {"layer": layer, "k": tensor_sha256(keys), "v": tensor_sha256(values)}
        for layer, (keys, values) in enumerate(snapshot)
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def _validate_snapshots(
    fresh_boundary: Snapshot,
    selected_sources: Mapping[str, Snapshot],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = validate_descriptor(descriptor)
    _require(isinstance(selected_sources, Mapping)
             and set(selected_sources) == set(SOURCE_HISTORIES),
             "bundle selected source set differs")
    fresh = _snapshot_geometry(
        fresh_boundary, "fresh boundary", max_tokens=MAX_FRESH_BOUNDARY_TOKENS)
    selected_width = (
        frozen["destination"]["r2_end"] - frozen["destination"]["r1_start"])
    _require(fresh["tokens"] == frozen["fresh_boundary_token_count"],
             "fresh boundary token count differs from descriptor")
    declared = frozen["cache_geometry"]
    for observed_key, declared_key in (
        ("layers", "layers"), ("batch", "batch"),
        ("kv_heads", "kv_heads"), ("head_dim", "head_dim"),
        ("dtype", "dtype"),
    ):
        _require(fresh[observed_key] == declared[declared_key],
                 "fresh cache geometry differs from descriptor")
    for history in SOURCE_HISTORIES:
        source = _snapshot_geometry(
            selected_sources[history], f"{history} selected R2",
            max_tokens=MAX_SELECTED_ROWS)
        _require(source["tokens"] == selected_width,
                 f"{history} selected width differs")
        for key in ("layers", "batch", "kv_heads", "head_dim", "dtype"):
            _require(source[key] == fresh[key],
                     f"{history} selected {key} differs from fresh")
        _require(snapshot_sha256(selected_sources[history]) ==
                 frozen["source"][history]["selected_rows_sha256"],
                 f"{history} selected-row descriptor hash differs")
    return frozen


def _tensor_key(source: str, layer: int, channel: str) -> str:
    return f"{source}.layer_{layer:03d}.{channel}"


def _tensors_from_snapshots(
    fresh_boundary: Snapshot,
    selected_sources: Mapping[str, Snapshot],
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for source, snapshot in (("F", fresh_boundary), *selected_sources.items()):
        for layer, (keys, values) in enumerate(snapshot):
            tensors[_tensor_key(source, layer, "k")] = (
                keys.detach().contiguous().cpu())
            tensors[_tensor_key(source, layer, "v")] = (
                values.detach().contiguous().cpu())
    return tensors


def _tensor_index(tensors: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": tensor_sha256(tensor),
        }
        for name, tensor in sorted(tensors.items())
    ]


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor],
                        metadata: Mapping[str, str]) -> None:
    _require(not path.exists(), f"refusing to overwrite bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        save_file(dict(tensors), str(temporary), metadata=dict(metadata))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise V13BundleError(f"bundle appeared during write: {path}") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def save_bundle(
    path: Path,
    *,
    fresh_boundary: Snapshot,
    selected_sources: Mapping[str, Snapshot],
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Exclusive-write one CPU bf16 bundle and return its outer binding."""

    frozen = _validate_snapshots(fresh_boundary, selected_sources, descriptor)
    tensors = _tensors_from_snapshots(fresh_boundary, selected_sources)
    index = _tensor_index(tensors)
    payload_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in tensors.values())
    _require(payload_bytes <= MAX_BUNDLE_BYTES - (1 << 20),
             "bundle tensor payload exceeds frozen byte bound")
    manifest = {
        "descriptor": frozen,
        "tensor_index": index,
        "tensor_index_sha256": sha256_bytes(canonical_json_bytes(index)),
    }
    _atomic_safetensors(path, tensors, {
        "schema": SCHEMA,
        "manifest": canonical_json_bytes(manifest).decode("utf-8"),
    })
    _require(path.stat().st_size <= MAX_BUNDLE_BYTES,
             "serialized bundle exceeds frozen byte bound")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
    }


def _snapshot_from_tensors(
    tensors: Mapping[str, torch.Tensor], source: str,
) -> Snapshot:
    return [
        (tensors[_tensor_key(source, layer, "k")].clone(),
         tensors[_tensor_key(source, layer, "v")].clone())
        for layer in range(REQUIRED_LAYERS)
    ]


def _load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise V13BundleError(f"cannot inspect bundle metadata: {exc}") from exc
    _require(metadata.get("schema") == SCHEMA
             and isinstance(metadata.get("manifest"), str),
             "bundle metadata schema/manifest differs")
    try:
        manifest = json.loads(metadata["manifest"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise V13BundleError(f"bundle manifest is invalid JSON: {exc}") from exc
    _require(isinstance(manifest, dict) and set(manifest) == {
        "descriptor", "tensor_index", "tensor_index_sha256",
    }, "bundle manifest field set differs")
    _require(canonical_json_bytes(manifest).decode("utf-8") == metadata["manifest"],
             "bundle manifest metadata is not canonical")
    tensors = load_file(str(path), device="cpu")
    index = _tensor_index(tensors)
    _require(index == manifest["tensor_index"]
             and manifest["tensor_index_sha256"] == sha256_bytes(
                 canonical_json_bytes(index)),
             "bundle tensor index/hash differs")
    expected_names = {
        _tensor_key(source, layer, channel)
        for source in HISTORIES
        for layer in range(REQUIRED_LAYERS)
        for channel in ("k", "v")
    }
    _require(set(tensors) == expected_names,
             "bundle tensor name/layer coverage differs")
    return manifest, tensors


def load_verified_bundle(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Load on CPU only after outer file and descriptor commitments agree."""

    _sha(expected_file_sha256, "expected bundle file SHA")
    _require(path.is_file() and path.stat().st_size <= MAX_BUNDLE_BYTES,
             "bundle is absent or exceeds frozen byte bound")
    _require(file_sha256(path) == expected_file_sha256,
             "bundle outer file hash differs")
    expected = validate_descriptor(expected_descriptor)
    manifest, tensors = _load_manifest(path)
    _require(validate_descriptor(manifest["descriptor"]) == expected,
             "bundle descriptor differs from external commitment")
    fresh = _snapshot_from_tensors(tensors, "F")
    sources = {
        history: _snapshot_from_tensors(tensors, history)
        for history in SOURCE_HISTORIES
    }
    _validate_snapshots(fresh, sources, expected)
    _require(all(tensor.device.type == "cpu" for tensor in tensors.values()),
             "loaded audit bundle is not CPU-resident")
    return {
        "path": str(path),
        "sha256": expected_file_sha256,
        "manifest": deepcopy(manifest),
        "tensors": tensors,
        "fresh": fresh,
        "sources": sources,
        "verified": True,
        "materialized": False,
    }


def _revalidate_loaded(value: Mapping[str, Any]) -> tuple[
    dict[str, Any], Snapshot, dict[str, Snapshot]
]:
    _require(isinstance(value, Mapping) and value.get("verified") is True
             and isinstance(value.get("manifest"), Mapping)
             and isinstance(value.get("tensors"), Mapping),
             "loaded bundle is not externally verified")
    manifest = value["manifest"]
    descriptor = validate_descriptor(manifest["descriptor"])
    tensors = value["tensors"]
    _require(_tensor_index(tensors) == manifest["tensor_index"],
             "loaded bundle tensors changed after verification")
    fresh = _snapshot_from_tensors(tensors, "F")
    sources = {
        history: _snapshot_from_tensors(tensors, history)
        for history in SOURCE_HISTORIES
    }
    _validate_snapshots(fresh, sources, descriptor)
    return descriptor, fresh, sources


def materialize_verified_bundle(
    loaded: Mapping[str, Any],
    *,
    device: torch.device | str,
    require_accelerator: bool = True,
) -> dict[str, Any]:
    """Explicitly copy one revalidated CPU foundation to an execution device."""

    descriptor, fresh, sources = _revalidate_loaded(loaded)
    target = torch.device(device)
    if require_accelerator:
        _require(target.type == "cuda",
                 "production bundle materialization requires a CUDA device")
    before = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", fresh), *sources.items())
    }

    def move(snapshot: Snapshot) -> Snapshot:
        return [
            (keys.to(device=target).clone().contiguous(),
             values.to(device=target).clone().contiguous())
            for keys, values in snapshot
        ]

    moved_fresh = move(fresh)
    moved_sources = {history: move(snapshot)
                     for history, snapshot in sources.items()}
    after = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", moved_fresh), *moved_sources.items())
    }
    _require(before == after, "CPU-to-device transfer changed bundle bits")
    return {
        "descriptor": descriptor,
        "fresh": moved_fresh,
        "sources": moved_sources,
        "device": str(target),
        "transfer_hashes": before,
        "verified": True,
        "materialized": True,
    }


def _revalidate_materialized(value: Mapping[str, Any]) -> tuple[
    dict[str, Any], Snapshot, dict[str, Snapshot]
]:
    _require(isinstance(value, Mapping) and value.get("verified") is True
             and value.get("materialized") is True,
             "bundle was not explicitly materialized")
    descriptor = validate_descriptor(value.get("descriptor", {}))
    fresh = value.get("fresh")
    sources = value.get("sources")
    _require(isinstance(fresh, list) and isinstance(sources, Mapping),
             "materialized bundle snapshots are missing")
    device_types = {
        tensor.device.type
        for snapshot in (fresh, *sources.values())
        for pair in snapshot for tensor in pair
    }
    _require(len(device_types) == 1,
             "materialized bundle spans multiple device types")
    transfer_hashes = value.get("transfer_hashes")
    _require(isinstance(transfer_hashes, Mapping)
             and set(transfer_hashes) == set(HISTORIES),
             "materialized bundle transfer hashes are missing")
    observed_hashes = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", fresh), *sources.items())
    }
    _require(observed_hashes == dict(transfer_hashes),
             "materialized bundle changed after device transfer")
    # Validate exact geometry/hashes on CPU copies without mutating the live
    # tensors; this makes every reconstruction begin from bound immutable bits.
    cpu_fresh = [(keys.detach().cpu(), values.detach().cpu())
                 for keys, values in fresh]
    cpu_sources = {
        history: [(keys.detach().cpu(), values.detach().cpu())
                  for keys, values in snapshot]
        for history, snapshot in sources.items()
    }
    _validate_snapshots(cpu_fresh, cpu_sources, descriptor)
    return descriptor, fresh, dict(sources)


def placebo_class_values(
    materialized: Mapping[str, Any],
) -> dict[str, dict[str, torch.Tensor]]:
    """Adapt F/C/W selected V rows to placebo ``[L,T,H,D]`` classes."""

    descriptor, fresh, sources = _revalidate_materialized(materialized)
    start = descriptor["destination"]["r1_start"]
    content_width = descriptor["destination"]["r1_end"] - start
    selected_width = descriptor["destination"]["r2_end"] - start

    def values_for(history: str) -> torch.Tensor:
        if history == "F":
            rows = [values[..., start:start + selected_width, :]
                    for _keys, values in fresh]
        else:
            rows = [values for _keys, values in sources[history]]
        return torch.stack(
            [row.squeeze(0).permute(1, 0, 2).contiguous() for row in rows],
            dim=0)

    result: dict[str, dict[str, torch.Tensor]] = {
        "fresh": {}, "correct": {}, "wrong": {},
    }
    for output_name, history in (("fresh", "F"), ("correct", "C"),
                                 ("wrong", "W")):
        selected = values_for(history)
        result[output_name]["content"] = selected[:, :content_width].clone()
        result[output_name]["structural"] = selected[:, content_width:].clone()
    return result


def reconstruct_arm_boundary(
    materialized: Mapping[str, Any],
    *,
    arm: str,
    placebo_values: Mapping[str, torch.Tensor] | None = None,
) -> tuple[Snapshot, dict[str, Any]]:
    """Reconstruct one serial arm from a fresh clone without continuation."""

    _require(arm in PRIMARY_ARMS, "arm differs from frozen primary order")
    descriptor, fresh, sources = _revalidate_materialized(materialized)
    start = descriptor["destination"]["r1_start"]
    content_width = descriptor["destination"]["r1_end"] - start
    end = descriptor["destination"]["r2_end"]
    width = end - start
    source_by_arm = {
        "FF": ("F", "F"),
        "CC": ("C", "C"),
        "WW": ("W", "W"),
        "FC": ("F", "C"),
        "FW": ("F", "W"),
        "VP": ("F", "VP"),
    }
    key_source, value_source = source_by_arm[arm]
    if arm == "VP":
        _require(isinstance(placebo_values, Mapping)
                 and set(placebo_values) == {"content", "structural"},
                 "VP arm requires both placebo value classes")
        content = placebo_values["content"]
        structural = placebo_values["structural"]
        _require(isinstance(content, torch.Tensor)
                 and isinstance(structural, torch.Tensor)
                 and content.shape[0] == structural.shape[0] == REQUIRED_LAYERS
                 and content.shape[1] == content_width
                 and structural.shape[1] == width - content_width,
                 "VP class tensor geometry differs")
        vp = torch.cat((content, structural), dim=1)
    else:
        _require(placebo_values is None,
                 "non-VP arm received placebo values")
        vp = None

    output: Snapshot = []
    evidence: list[dict[str, Any]] = []
    for layer in range(REQUIRED_LAYERS):
        fresh_k, fresh_v = fresh[layer]
        after_k, after_v = fresh_k.clone(), fresh_v.clone()
        if key_source != "F":
            after_k[..., start:end, :] = sources[key_source][layer][0]
        if value_source in SOURCE_HISTORIES:
            after_v[..., start:end, :] = sources[value_source][layer][1]
        elif value_source == "VP":
            assert vp is not None
            rows = vp[layer].permute(1, 0, 2).unsqueeze(0).contiguous()
            _require(rows.dtype == after_v.dtype and rows.device == after_v.device,
                     "VP values differ in dtype/device")
            after_v[..., start:end, :] = rows
        output.append((after_k, after_v))
        row = {
            "layer": layer,
            "before_k_sha256": tensor_sha256(fresh_k),
            "before_v_sha256": tensor_sha256(fresh_v),
            "after_k_sha256": tensor_sha256(after_k),
            "after_v_sha256": tensor_sha256(after_v),
            "outside_prefix_k_before_sha256": tensor_sha256(
                fresh_k[..., :start, :]),
            "outside_prefix_k_after_sha256": tensor_sha256(
                after_k[..., :start, :]),
            "outside_prefix_v_before_sha256": tensor_sha256(
                fresh_v[..., :start, :]),
            "outside_prefix_v_after_sha256": tensor_sha256(
                after_v[..., :start, :]),
        }
        _require(row["outside_prefix_k_before_sha256"] ==
                 row["outside_prefix_k_after_sha256"]
                 and row["outside_prefix_v_before_sha256"] ==
                 row["outside_prefix_v_after_sha256"],
                 f"arm {arm} layer {layer} changed state before R2 destination")
        evidence.append(row)
    return output, {
        "arm": arm,
        "key_source": key_source,
        "value_source": value_source,
        "destination_start": start,
        "destination_end": end,
        "selected_row_count": width,
        "per_layer": evidence,
        "output_sha256": snapshot_sha256(output),
    }


__all__ = [
    "MAX_BUNDLE_BYTES",
    "MAX_FRESH_BOUNDARY_TOKENS",
    "MAX_SELECTED_ROWS",
    "SCHEMA",
    "V13BundleError",
    "build_descriptor",
    "canonical_json_bytes",
    "file_sha256",
    "load_verified_bundle",
    "materialize_verified_bundle",
    "placebo_class_values",
    "reconstruct_arm_boundary",
    "save_bundle",
    "sha256_bytes",
    "snapshot_sha256",
    "validate_descriptor",
]
