"""Bounded lossless selected-state bundles for powered successor v13.

One bundle contains a fresh cache boundary through dynamic R2 and exact C/W
selected R2 rows for one case/render/schedule.  Files load on CPU for audit.
Treatment must reload the externally bound file through
:func:`materialize_verified_bundle` onto real CUDA before any continuation.
VP is admitted only through a Phase-A receipt that treatment recomputes from
the bound F/C/W rows.  The module contains no model forward or outcome logic.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from coherent_canary_runtime import Snapshot, tensor_sha256
from powered_v13_placebo import build_value_row_placebo
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
MAX_LIVE_CACHE_TOKENS = 7000
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 64 * 1024
MAX_METADATA_BYTES = 1 * 1024 * 1024
MAX_IDENTIFIER_BYTES = 128
TECHNICAL_CASE_IDS = ("technical_e01", "technical_long")
VP_RECEIPT_SCHEMA = "coherent-state-powered-successor-v13-vp-receipt-v1"
_CUDA_AUTHORIZATION = "POWERED_V13_CUDA_EXECUTION_V1"
_CPU_TEST_AUTHORIZATION = "POWERED_V13_PRIVATE_CPU_TEST_ONLY_V1"
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
    _require(len(value.encode("utf-8")) <= MAX_IDENTIFIER_BYTES,
             f"{label} exceeds {MAX_IDENTIFIER_BYTES} bytes")
    return value


def _case_id(value: object) -> str:
    _require(isinstance(value, str), "case_id is not a string")
    _require(_SHA256.fullmatch(value) is not None or value in TECHNICAL_CASE_IDS,
             "case_id is neither a stable 64-hex ID nor a technical allowlist ID")
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


def _json_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _plain_int_sequence(value: object, label: str, *, length: int,
                        maximum: int) -> tuple[int, ...]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
             f"{label} is not an integer sequence")
    result = tuple(value)
    _require(len(result) == length, f"{label} width differs")
    _require(all(isinstance(item, int) and not isinstance(item, bool)
                 and 0 <= item <= maximum for item in result),
             f"{label} contains an invalid integer")
    return result


def validate_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "design_id", "release_sha256",
        "runtime_fingerprint_sha256", "primary_batch_id", "gpu_uuid",
        "case_id", "stable_candidate_id", "render_id", "schedule",
        "history_ids", "state_keys",
        "plan_sha256s", "foundation_bindings", "selected_map_sha256",
        "selected_token_ids", "selected_token_ids_sha256",
        "logical_position_ids", "logical_position_ids_sha256",
        "fresh_boundary_token_count", "destination", "source",
        "capture_bindings", "cache_geometry",
    }
    _require(isinstance(value, Mapping) and set(value) == expected,
             "bundle descriptor field set differs")
    _require(value.get("schema") == SCHEMA and value.get("design_id") == DESIGN_ID,
             "bundle descriptor schema/design differs")
    _sha(value.get("release_sha256"), "release_sha256")
    _sha(value.get("runtime_fingerprint_sha256"), "runtime_fingerprint_sha256")
    _slug(value.get("primary_batch_id"), "primary_batch_id")
    _slug(value.get("gpu_uuid"), "gpu_uuid")
    case_id = _case_id(value.get("case_id"))
    stable_candidate_id = _sha(
        value.get("stable_candidate_id"), "stable_candidate_id")
    if case_id not in TECHNICAL_CASE_IDS:
        _require(stable_candidate_id == case_id,
                 "semantic case_id differs from stable_candidate_id")
    _require(value.get("render_id") in ("r1", "r2"),
             "render_id must be r1 or r2")
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
    _sha(value.get("logical_position_ids_sha256"),
         "logical_position_ids_sha256")

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
    selected_token_ids = _plain_int_sequence(
        value.get("selected_token_ids"), "descriptor selected token IDs",
        length=end - start, maximum=(1 << 31) - 1)
    logical_position_ids = _plain_int_sequence(
        value.get("logical_position_ids"), "descriptor logical position IDs",
        length=end - start, maximum=MAX_LIVE_CACHE_TOKENS - 1)
    _require(logical_position_ids == tuple(range(
        logical_position_ids[0], logical_position_ids[0] + end - start)),
        "descriptor logical R2 positions are not contiguous")
    _require(_json_sha256(list(selected_token_ids)) ==
             value["selected_token_ids_sha256"],
             "descriptor selected-token IDs/hash differ")
    _require(_json_sha256(list(logical_position_ids)) ==
             value["logical_position_ids_sha256"],
             "descriptor logical-position IDs/hash differ")

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

    captures = value.get("capture_bindings")
    _require(isinstance(captures, Mapping) and set(captures) == set(HISTORIES),
             "bundle capture history fields differ")
    common_interval: tuple[int, int, int] | None = None
    for history in HISTORIES:
        capture = captures[history]
        _require(isinstance(capture, Mapping) and set(capture) == {
            "history", "schedule", "plan_sha256", "logical_start",
            "logical_r1_end", "logical_end", "logical_position_ids_sha256",
            "selected_token_ids_sha256",
        }, f"bundle {history} capture fields differ")
        logical_start = capture.get("logical_start")
        logical_r1_end = capture.get("logical_r1_end")
        logical_end = capture.get("logical_end")
        _require(
            capture.get("history") == history
            and capture.get("schedule") == value["schedule"]
            and capture.get("plan_sha256") == plans[history],
            f"bundle {history} capture origin differs")
        _require(
            isinstance(logical_start, int) and not isinstance(logical_start, bool)
            and isinstance(logical_r1_end, int)
            and not isinstance(logical_r1_end, bool)
            and isinstance(logical_end, int) and not isinstance(logical_end, bool)
            and 0 <= logical_start < logical_r1_end < logical_end
            and logical_end <= MAX_LIVE_CACHE_TOKENS,
            f"bundle {history} logical R1/R2 geometry differs")
        interval = (logical_start, logical_r1_end, logical_end)
        if common_interval is None:
            common_interval = interval
        _require(interval == common_interval,
                 "bundle C/W/F logical R2 intervals differ")
        _require(logical_end - logical_start == end - start
                 and logical_r1_end - logical_start == content_end - start,
                 f"bundle {history} logical/physical R1/R2 widths differ")
        _require(logical_start == logical_position_ids[0]
                 and logical_end == logical_position_ids[-1] + 1,
                 f"bundle {history} interval/position IDs differ")
        _require(capture.get("logical_position_ids_sha256") ==
                 value["logical_position_ids_sha256"],
                 f"bundle {history} logical-position hash differs")
        _require(capture.get("selected_token_ids_sha256") ==
                 value["selected_token_ids_sha256"],
                 f"bundle {history} selected-token hash differs")
        if history in SOURCE_HISTORIES:
            _require(
                source[history]["logical_start"] == logical_start
                and source[history]["logical_end"] == logical_end,
                f"bundle {history} source/capture interval differs")

    state_keys = value.get("state_keys")
    _require(isinstance(state_keys, Mapping) and set(state_keys) == set(HISTORIES),
             "bundle state-key histories differ")
    for history in HISTORIES:
        _require(state_keys[history] == _state_key(value, history),
                 f"bundle {history} state key differs")
    frozen = deepcopy(dict(value))
    _require(len(canonical_json_bytes(frozen)) <= MAX_DESCRIPTOR_BYTES,
             "bundle descriptor exceeds frozen byte bound")
    return frozen


def build_descriptor(
    *,
    release_sha256: str,
    runtime_fingerprint_sha256: str,
    primary_batch_id: str,
    gpu_uuid: str,
    case_id: str,
    stable_candidate_id: str | None = None,
    render_id: str,
    schedule: str,
    plan_sha256s: Mapping[str, str],
    foundation_bindings: Mapping[str, str],
    selected_map_sha256: str,
    selected_token_ids: Mapping[str, Sequence[int]],
    logical_position_ids: Mapping[str, Sequence[int]],
    fresh_boundary_token_count: int,
    r1_start: int,
    r1_end: int,
    r2_end: int,
    source_intervals: Mapping[str, tuple[int, int]],
    source_rows_sha256: Mapping[str, str],
    kv_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    _require(isinstance(selected_token_ids, Mapping)
             and set(selected_token_ids) == set(HISTORIES),
             "selected token-ID histories differ")
    _require(isinstance(logical_position_ids, Mapping)
             and set(logical_position_ids) == set(HISTORIES),
             "logical position-ID histories differ")
    _require(isinstance(source_intervals, Mapping)
             and set(source_intervals) == set(SOURCE_HISTORIES),
             "source interval histories differ")
    _require(isinstance(source_rows_sha256, Mapping)
             and set(source_rows_sha256) == set(SOURCE_HISTORIES),
             "source row-hash histories differ")
    _require(all(isinstance(point, int) and not isinstance(point, bool)
                 for point in (r1_start, r1_end, r2_end))
             and 0 <= r1_start < r1_end < r2_end,
             "destination R1/R2 points differ")
    selected_width = r2_end - r1_start
    token_rows = {
        history: _plain_int_sequence(
            selected_token_ids[history], f"{history} selected token IDs",
            length=selected_width, maximum=(1 << 31) - 1)
        for history in HISTORIES
    }
    position_rows = {
        history: _plain_int_sequence(
            logical_position_ids[history], f"{history} logical position IDs",
            length=selected_width, maximum=MAX_LIVE_CACHE_TOKENS - 1)
        for history in HISTORIES
    }
    _require(len(set(token_rows.values())) == 1,
             "bundle C/W/F selected token IDs differ")
    _require(len(set(position_rows.values())) == 1,
             "bundle C/W/F logical position IDs differ")
    common_tokens = token_rows["F"]
    common_positions = position_rows["F"]
    _require(common_positions == tuple(range(
        common_positions[0], common_positions[0] + selected_width)),
        "bundle logical R2 positions are not contiguous")
    logical_start = common_positions[0]
    logical_end = logical_start + selected_width
    logical_r1_end = logical_start + (r1_end - r1_start)
    for history in SOURCE_HISTORIES:
        interval = source_intervals[history]
        _require(isinstance(interval, Sequence) and len(interval) == 2
                 and tuple(interval) == (logical_start, logical_end),
                 "bundle C/W/F logical R2 intervals differ")
    selected_token_ids_sha256 = _json_sha256(list(common_tokens))
    logical_position_ids_sha256 = _json_sha256(list(common_positions))
    resolved_stable_id = stable_candidate_id or case_id
    descriptor: dict[str, Any] = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "release_sha256": release_sha256,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "primary_batch_id": primary_batch_id,
        "gpu_uuid": gpu_uuid,
        "case_id": case_id,
        "stable_candidate_id": resolved_stable_id,
        "render_id": render_id,
        "schedule": schedule,
        "history_ids": list(HISTORIES),
        "state_keys": {},
        "plan_sha256s": dict(plan_sha256s),
        "foundation_bindings": dict(foundation_bindings),
        "selected_map_sha256": selected_map_sha256,
        "selected_token_ids": list(common_tokens),
        "selected_token_ids_sha256": selected_token_ids_sha256,
        "logical_position_ids": list(common_positions),
        "logical_position_ids_sha256": logical_position_ids_sha256,
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
        "capture_bindings": {
            history: {
                "history": history,
                "schedule": schedule,
                "plan_sha256": plan_sha256s[history],
                "logical_start": logical_start,
                "logical_r1_end": logical_r1_end,
                "logical_end": logical_end,
                "logical_position_ids_sha256": logical_position_ids_sha256,
                "selected_token_ids_sha256": selected_token_ids_sha256,
            }
            for history in HISTORIES
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


def _snapshot_geometry(snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]], label: str,
                       *, max_tokens: int) -> dict[str, int | str]:
    _require(isinstance(snapshot, (list, tuple))
             and len(snapshot) == REQUIRED_LAYERS,
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


def snapshot_sha256(
    snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> str:
    rows = [
        {"layer": layer, "k": tensor_sha256(keys), "v": tensor_sha256(values)}
        for layer, (keys, values) in enumerate(snapshot)
    ]
    return sha256_bytes(canonical_json_bytes(rows))


def _validate_snapshots(
    fresh_boundary: Sequence[tuple[torch.Tensor, torch.Tensor]],
    selected_sources: Mapping[
        str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
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
    fresh_boundary: Sequence[tuple[torch.Tensor, torch.Tensor]],
    selected_sources: Mapping[
        str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
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
                        metadata: Mapping[str, str], *, max_bytes: int) -> None:
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
        _require(temporary.stat().st_size <= max_bytes,
                 "serialized bundle exceeds frozen byte bound before publish")
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
    manifest_bytes = canonical_json_bytes(manifest)
    _require(len(manifest_bytes) <= MAX_METADATA_BYTES,
             "bundle manifest exceeds frozen metadata bound")
    _atomic_safetensors(path, tensors, {
        "schema": SCHEMA,
        "manifest": manifest_bytes.decode("utf-8"),
    }, max_bytes=MAX_BUNDLE_BYTES)
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
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (tensors[_tensor_key(source, layer, "k")].clone(),
         tensors[_tensor_key(source, layer, "v")].clone())
        for layer in range(REQUIRED_LAYERS)
    )


FrozenSnapshot = tuple[tuple[torch.Tensor, torch.Tensor], ...]


def _snapshot_mapping(
    value: Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
) -> Mapping[str, FrozenSnapshot]:
    return MappingProxyType({
        history: tuple((keys, values) for keys, values in value[history])
        for history in SOURCE_HISTORIES
    })


def _verified_anchor_sha256(
    *, file_sha256: str, descriptor_bytes: bytes, manifest_bytes: bytes,
    tensor_index_bytes: bytes,
) -> str:
    return _json_sha256([
        SCHEMA,
        file_sha256,
        sha256_bytes(descriptor_bytes),
        sha256_bytes(manifest_bytes),
        sha256_bytes(tensor_index_bytes),
    ])


@dataclass(frozen=True)
class VerifiedBundle:
    """CPU-only audit handle whose external commitments are immutable bytes."""

    path: str
    sha256: str
    descriptor_bytes: bytes = field(repr=False)
    manifest_bytes: bytes = field(repr=False)
    tensor_index_bytes: bytes = field(repr=False)
    tensor_index_sha256: str
    manifest_sha256: str
    anchor_sha256: str
    fresh: FrozenSnapshot = field(repr=False, compare=False)
    sources: Mapping[str, FrozenSnapshot] = field(repr=False, compare=False)

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(self.descriptor_bytes)

    @property
    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_bytes)

    @property
    def tensor_index(self) -> list[dict[str, Any]]:
        return json.loads(self.tensor_index_bytes)

    @property
    def verified(self) -> bool:
        return True

    @property
    def materialized(self) -> bool:
        return False


def _load_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], bytes]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise V13BundleError(f"cannot inspect bundle metadata: {exc}") from exc
    _require(metadata.get("schema") == SCHEMA
             and isinstance(metadata.get("manifest"), str),
             "bundle metadata schema/manifest differs")
    _require(len(metadata["manifest"].encode("utf-8")) <= MAX_METADATA_BYTES,
             "bundle manifest exceeds frozen metadata bound")
    try:
        manifest = json.loads(metadata["manifest"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise V13BundleError(f"bundle manifest is invalid JSON: {exc}") from exc
    _require(isinstance(manifest, dict) and set(manifest) == {
        "descriptor", "tensor_index", "tensor_index_sha256",
    }, "bundle manifest field set differs")
    _require(canonical_json_bytes(manifest).decode("utf-8") == metadata["manifest"],
             "bundle manifest metadata is not canonical")
    manifest_bytes = metadata["manifest"].encode("utf-8")
    try:
        tensors = load_file(str(path), device="cpu")
    except Exception as exc:
        raise V13BundleError(f"cannot load bundle tensors: {exc}") from exc
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
    return manifest, tensors, manifest_bytes


def load_verified_bundle(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_descriptor: Mapping[str, Any],
) -> VerifiedBundle:
    """Load on CPU only after outer file and descriptor commitments agree."""

    _sha(expected_file_sha256, "expected bundle file SHA")
    _require(isinstance(path, (str, os.PathLike)), "bundle path is not path-like")
    path = Path(path)
    _require(not path.is_symlink() and path.is_file()
             and path.stat().st_size <= MAX_BUNDLE_BYTES,
             "bundle is absent, symlinked, or exceeds frozen byte bound")
    stat_before = path.stat()
    hash_before = file_sha256(path)
    _require(hash_before == expected_file_sha256,
             "bundle outer file hash differs")
    expected = validate_descriptor(expected_descriptor)
    expected_descriptor_bytes = canonical_json_bytes(expected)
    manifest, tensors, manifest_bytes = _load_manifest(path)
    stat_after = path.stat()
    hash_after = file_sha256(path)
    _require(
        (stat_before.st_dev, stat_before.st_ino, stat_before.st_size,
         stat_before.st_mtime_ns) ==
        (stat_after.st_dev, stat_after.st_ino, stat_after.st_size,
         stat_after.st_mtime_ns)
        and hash_before == hash_after == expected_file_sha256,
        "bundle changed while it was being read")
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
    tensor_index_bytes = canonical_json_bytes(manifest["tensor_index"])
    anchor = _verified_anchor_sha256(
        file_sha256=expected_file_sha256,
        descriptor_bytes=expected_descriptor_bytes,
        manifest_bytes=manifest_bytes,
        tensor_index_bytes=tensor_index_bytes)
    return VerifiedBundle(
        path=str(path.resolve()),
        sha256=expected_file_sha256,
        descriptor_bytes=expected_descriptor_bytes,
        manifest_bytes=manifest_bytes,
        tensor_index_bytes=tensor_index_bytes,
        tensor_index_sha256=manifest["tensor_index_sha256"],
        manifest_sha256=sha256_bytes(manifest_bytes),
        anchor_sha256=anchor,
        fresh=tuple(fresh),
        sources=_snapshot_mapping(sources),
    )


def _revalidate_loaded(value: VerifiedBundle) -> tuple[
    dict[str, Any], FrozenSnapshot, dict[str, FrozenSnapshot]
]:
    _require(isinstance(value, VerifiedBundle),
             "loaded bundle is not an immutable verified audit handle")
    _sha(value.sha256, "verified bundle file SHA")
    descriptor = validate_descriptor(json.loads(value.descriptor_bytes))
    _require(canonical_json_bytes(descriptor) == value.descriptor_bytes,
             "verified descriptor anchor changed")
    try:
        manifest = json.loads(value.manifest_bytes)
        tensor_index = json.loads(value.tensor_index_bytes)
    except (json.JSONDecodeError, TypeError) as exc:
        raise V13BundleError(f"verified immutable anchor is invalid JSON: {exc}") from exc
    _require(isinstance(manifest, dict) and set(manifest) == {
        "descriptor", "tensor_index", "tensor_index_sha256"},
        "verified manifest field set differs")
    _require(canonical_json_bytes(manifest) == value.manifest_bytes
             and validate_descriptor(manifest["descriptor"]) == descriptor,
             "verified manifest/descriptor anchor differs")
    _require(isinstance(tensor_index, list)
             and canonical_json_bytes(tensor_index) == value.tensor_index_bytes
             and manifest["tensor_index"] == tensor_index
             and manifest["tensor_index_sha256"] == value.tensor_index_sha256
             == sha256_bytes(value.tensor_index_bytes),
             "verified tensor-index anchor differs")
    sources = dict(value.sources)
    tensors = _tensors_from_snapshots(value.fresh, sources)
    _require(_tensor_index(tensors) == tensor_index,
             "loaded bundle tensors changed after verification")
    _validate_snapshots(value.fresh, sources, descriptor)
    _require(value.manifest_sha256 == sha256_bytes(value.manifest_bytes)
             and value.anchor_sha256 == _verified_anchor_sha256(
                 file_sha256=value.sha256,
                 descriptor_bytes=value.descriptor_bytes,
                 manifest_bytes=value.manifest_bytes,
                 tensor_index_bytes=value.tensor_index_bytes),
             "verified external anchor differs")
    return descriptor, value.fresh, sources


def _materialized_seal(
    *, authorization: str, file_sha256: str, descriptor_bytes: bytes,
    manifest_sha256: str, audit_anchor_sha256: str, device: str,
    transfer_hashes: tuple[tuple[str, str], ...],
) -> str:
    return _json_sha256([
        SCHEMA, authorization, file_sha256, sha256_bytes(descriptor_bytes),
        manifest_sha256, audit_anchor_sha256, device,
        [list(row) for row in transfer_hashes],
    ])


@dataclass(frozen=True)
class _MaterializedBundle:
    path: str
    file_sha256: str
    descriptor_bytes: bytes = field(repr=False)
    manifest_sha256: str
    audit_anchor_sha256: str
    authorization: str
    device: str
    transfer_hashes: tuple[tuple[str, str], ...]
    seal_sha256: str
    fresh: FrozenSnapshot = field(repr=False, compare=False)
    sources: Mapping[str, FrozenSnapshot] = field(repr=False, compare=False)

    @property
    def descriptor(self) -> dict[str, Any]:
        return json.loads(self.descriptor_bytes)

    @property
    def verified(self) -> bool:
        return True

    @property
    def materialized(self) -> bool:
        return True


def _materialize_loaded(
    loaded: VerifiedBundle, *, target: torch.device, authorization: str,
) -> _MaterializedBundle:
    descriptor, fresh, sources = _revalidate_loaded(loaded)
    before = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", fresh), *sources.items())
    }

    def move(
        snapshot: Sequence[tuple[torch.Tensor, torch.Tensor]],
    ) -> FrozenSnapshot:
        return tuple(
            (keys.to(device=target).clone().contiguous(),
             values.to(device=target).clone().contiguous())
            for keys, values in snapshot)

    moved_fresh = move(fresh)
    moved_sources = {history: move(snapshot)
                     for history, snapshot in sources.items()}
    devices = {
        tensor.device
        for snapshot in (moved_fresh, *moved_sources.values())
        for pair in snapshot for tensor in pair
    }
    _require(len(devices) == 1, "materialized bundle spans multiple devices")
    actual_device = next(iter(devices))
    if authorization == _CUDA_AUTHORIZATION:
        _require(actual_device.type == "cuda",
                 "production bundle did not materialize on CUDA")
    else:
        _require(authorization == _CPU_TEST_AUTHORIZATION
                 and actual_device.type == "cpu",
                 "private test materialization authorization/device differs")
    after = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", moved_fresh), *moved_sources.items())
    }
    _require(before == after, "CPU-to-device transfer changed bundle bits")
    transfer_hashes = tuple((history, before[history]) for history in HISTORIES)
    device_name = str(actual_device)
    seal = _materialized_seal(
        authorization=authorization,
        file_sha256=loaded.sha256,
        descriptor_bytes=loaded.descriptor_bytes,
        manifest_sha256=loaded.manifest_sha256,
        audit_anchor_sha256=loaded.anchor_sha256,
        device=device_name,
        transfer_hashes=transfer_hashes)
    return _MaterializedBundle(
        path=loaded.path,
        file_sha256=loaded.sha256,
        descriptor_bytes=loaded.descriptor_bytes,
        manifest_sha256=loaded.manifest_sha256,
        audit_anchor_sha256=loaded.anchor_sha256,
        authorization=authorization,
        device=device_name,
        transfer_hashes=transfer_hashes,
        seal_sha256=seal,
        fresh=moved_fresh,
        sources=_snapshot_mapping(moved_sources),
    )


def materialize_verified_bundle(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_descriptor: Mapping[str, Any],
    device: torch.device | str,
) -> _MaterializedBundle:
    """Reload external commitments and authorize execution only on real CUDA."""

    _require(isinstance(path, (str, os.PathLike)),
             "production bundle path is not path-like")
    target = torch.device(device)
    _require(target.type == "cuda",
             "production bundle materialization requires a CUDA device")
    _require(torch.cuda.is_available(),
             "production bundle materialization requires available CUDA")
    loaded = load_verified_bundle(
        path, expected_file_sha256=expected_file_sha256,
        expected_descriptor=expected_descriptor)
    return _materialize_loaded(
        loaded, target=target, authorization=_CUDA_AUTHORIZATION)


def _materialize_verified_bundle_for_test(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_descriptor: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> _MaterializedBundle:
    """Private CPU-only unit-test path; public reconstruction rejects it."""

    target = torch.device(device)
    _require(target.type == "cpu", "private test materialization must remain CPU")
    loaded = load_verified_bundle(
        path, expected_file_sha256=expected_file_sha256,
        expected_descriptor=expected_descriptor)
    return _materialize_loaded(
        loaded, target=target, authorization=_CPU_TEST_AUTHORIZATION)


def _revalidate_materialized(
    value: _MaterializedBundle, *, require_cuda: bool,
) -> tuple[
    dict[str, Any], FrozenSnapshot, dict[str, FrozenSnapshot]
]:
    _require(isinstance(value, _MaterializedBundle),
             "bundle was not explicitly materialized")
    descriptor = validate_descriptor(json.loads(value.descriptor_bytes))
    _require(canonical_json_bytes(descriptor) == value.descriptor_bytes,
             "materialized descriptor anchor differs")
    for digest, label in (
        (value.file_sha256, "materialized file SHA"),
        (value.manifest_sha256, "materialized manifest SHA"),
        (value.audit_anchor_sha256, "materialized audit anchor SHA"),
    ):
        _sha(digest, label)
    transfer_hashes = tuple(value.transfer_hashes)
    _require(transfer_hashes == tuple(
        (history, dict(transfer_hashes).get(history)) for history in HISTORIES)
        and all(_SHA256.fullmatch(digest) is not None
                for _history, digest in transfer_hashes),
        "materialized transfer-hash anchor differs")
    _require(value.seal_sha256 == _materialized_seal(
        authorization=value.authorization,
        file_sha256=value.file_sha256,
        descriptor_bytes=value.descriptor_bytes,
        manifest_sha256=value.manifest_sha256,
        audit_anchor_sha256=value.audit_anchor_sha256,
        device=value.device,
        transfer_hashes=transfer_hashes),
        "materialized immutable seal differs")
    devices = {
        tensor.device
        for snapshot in (value.fresh, *value.sources.values())
        for pair in snapshot for tensor in pair
    }
    _require(len(devices) == 1, "materialized bundle spans multiple devices")
    actual_device = next(iter(devices))
    _require(str(actual_device) == value.device,
             "materialized device anchor differs")
    if require_cuda:
        _require(value.authorization == _CUDA_AUTHORIZATION
                 and actual_device.type == "cuda",
                 "public reconstruction requires CUDA-authorized materialization")
    else:
        _require(value.authorization in {
            _CUDA_AUTHORIZATION, _CPU_TEST_AUTHORIZATION},
            "materialized authorization differs")
    observed_hashes = {
        source: snapshot_sha256(snapshot)
        for source, snapshot in (("F", value.fresh), *value.sources.items())
    }
    _require(observed_hashes == dict(transfer_hashes),
             "materialized bundle changed after device transfer")
    # Validate exact geometry/hashes on CPU copies without mutating the live
    # tensors; this makes every reconstruction begin from bound immutable bits.
    cpu_fresh = [(keys.detach().cpu(), values.detach().cpu())
                 for keys, values in value.fresh]
    cpu_sources = {
        history: [(keys.detach().cpu(), values.detach().cpu())
                  for keys, values in snapshot]
        for history, snapshot in value.sources.items()
    }
    _validate_snapshots(cpu_fresh, cpu_sources, descriptor)
    return descriptor, value.fresh, dict(value.sources)


def _placebo_class_values_from_parts(
    descriptor: Mapping[str, Any],
    fresh: Sequence[tuple[torch.Tensor, torch.Tensor]],
    sources: Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Adapt F/C/W selected V rows to placebo ``[L,T,H,D]`` classes."""

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


def placebo_class_values(
    materialized: _MaterializedBundle,
) -> dict[str, dict[str, torch.Tensor]]:
    """Adapt a CUDA-authorized F/C/W bundle to ``[L,T,H,D]`` classes."""

    descriptor, fresh, sources = _revalidate_materialized(
        materialized, require_cuda=True)
    return _placebo_class_values_from_parts(descriptor, fresh, sources)


def _placebo_class_values_for_test(
    materialized: _MaterializedBundle,
) -> dict[str, dict[str, torch.Tensor]]:
    descriptor, fresh, sources = _revalidate_materialized(
        materialized, require_cuda=False)
    return _placebo_class_values_from_parts(descriptor, fresh, sources)


def _normalized_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": (
            "NaN" if math.isnan(value) else
            "+Infinity" if value > 0 else "-Infinity")}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise V13BundleError(
        f"placebo diagnostics contain unsupported {type(value).__name__}")


def _validate_sha_mapping(value: object, keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == keys,
             f"{label} fields differ")
    for key in keys:
        _sha(value[key], f"{label} {key}")


def _validate_vp_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "design_id", "bundle_file_sha256",
        "bundle_manifest_sha256", "descriptor_sha256", "case_id",
        "stable_candidate_id", "render_id", "schedule", "state_keys_sha256",
        "selected_map_sha256", "selected_token_ids_sha256",
        "logical_position_ids_sha256", "source_snapshot_sha256",
        "source_class_sha256", "status", "row_map", "result_sha256",
        "result_layer_sha256", "diagnostics_sha256",
    }
    _require(isinstance(value, Mapping) and set(value) == expected,
             "VP receipt field set differs")
    _require(value.get("schema") == VP_RECEIPT_SCHEMA
             and value.get("design_id") == DESIGN_ID,
             "VP receipt schema/design differs")
    for key in (
        "bundle_file_sha256", "bundle_manifest_sha256", "descriptor_sha256",
        "stable_candidate_id", "state_keys_sha256", "selected_map_sha256",
        "selected_token_ids_sha256", "logical_position_ids_sha256",
        "diagnostics_sha256",
    ):
        _sha(value.get(key), f"VP receipt {key}")
    _case_id(value.get("case_id"))
    _require(value.get("render_id") in ("r1", "r2"),
             "VP receipt render differs")
    _require(value.get("schedule") in SCHEDULES,
             "VP receipt schedule differs")
    _validate_sha_mapping(
        value.get("source_snapshot_sha256"), set(HISTORIES),
        "VP receipt source snapshot")
    classes = {"content", "structural"}
    class_hashes = value.get("source_class_sha256")
    _require(isinstance(class_hashes, Mapping)
             and set(class_hashes) == {"fresh", "correct", "wrong"},
             "VP receipt source-class histories differ")
    for source in class_hashes:
        _validate_sha_mapping(
            class_hashes[source], classes,
            f"VP receipt {source} source class")
    status = value.get("status")
    _require(status in {"AVAILABLE", "PLACEBO_UNAVAILABLE"},
             "VP receipt status differs")
    if status == "AVAILABLE":
        row_map = value.get("row_map")
        _require(isinstance(row_map, Mapping) and set(row_map) == classes,
                 "available VP receipt row map differs")
        for class_name in classes:
            rows = row_map[class_name]
            _require(isinstance(rows, list) and bool(rows)
                     and all(isinstance(row, Mapping)
                             and set(row) == {"destination", "donor"}
                             and all(isinstance(row[name], int)
                                     and not isinstance(row[name], bool)
                                     and row[name] >= 0
                                     for name in ("destination", "donor"))
                             for row in rows),
                     f"available VP receipt {class_name} row map differs")
        _validate_sha_mapping(
            value.get("result_sha256"), classes, "VP receipt result")
        layer_hashes = value.get("result_layer_sha256")
        _require(isinstance(layer_hashes, Mapping)
                 and set(layer_hashes) == classes,
                 "VP receipt result-layer classes differ")
        for class_name in classes:
            rows = layer_hashes[class_name]
            _require(isinstance(rows, list) and len(rows) == REQUIRED_LAYERS,
                     f"VP receipt {class_name} result-layer count differs")
            for layer, digest in enumerate(rows):
                _sha(digest, f"VP receipt {class_name} result layer {layer}")
    else:
        _require(value.get("row_map") is None
                 and value.get("result_sha256") is None
                 and value.get("result_layer_sha256") is None,
                 "unavailable VP receipt contains a selected result")
    frozen = deepcopy(dict(value))
    _require(len(canonical_json_bytes(frozen)) <= MAX_METADATA_BYTES,
             "VP receipt exceeds frozen metadata bound")
    return frozen


def _run_bound_placebo(
    *,
    file_sha256: str,
    manifest_sha256: str,
    descriptor_bytes: bytes,
    fresh: Sequence[tuple[torch.Tensor, torch.Tensor]],
    sources: Mapping[str, Sequence[tuple[torch.Tensor, torch.Tensor]]],
) -> tuple[dict[str, Any], str, dict[str, Any], Any]:
    descriptor = validate_descriptor(json.loads(descriptor_bytes))
    _require(canonical_json_bytes(descriptor) == descriptor_bytes,
             "VP descriptor anchor differs")
    classes = _placebo_class_values_from_parts(descriptor, fresh, sources)
    cpu_classes = {
        source: {
            class_name: tensor.detach().to(device="cpu").clone().contiguous()
            for class_name, tensor in class_values.items()
        }
        for source, class_values in classes.items()
    }
    result = build_value_row_placebo(
        cpu_classes["fresh"], cpu_classes["correct"], cpu_classes["wrong"],
        design_id=DESIGN_ID,
        stable_candidate_id=descriptor["stable_candidate_id"],
        render_id=descriptor["render_id"])
    diagnostics = _normalized_json(result.diagnostics)
    diagnostics_sha256 = _json_sha256(diagnostics)
    result_sha256 = None
    result_layer_sha256 = None
    if result.status == "AVAILABLE":
        _require(result.values is not None and result.row_map is not None,
                 "available selector result lacks values/map")
        result_sha256 = {
            class_name: tensor_sha256(result.values[class_name])
            for class_name in ("content", "structural")
        }
        result_layer_sha256 = {
            class_name: [
                tensor_sha256(result.values[class_name][layer])
                for layer in range(REQUIRED_LAYERS)
            ]
            for class_name in ("content", "structural")
        }
    else:
        _require(result.values is None and result.row_map is None,
                 "unavailable selector result contains values/map")
    source_class_sha256 = {
        source: {
            class_name: tensor_sha256(cpu_classes[source][class_name])
            for class_name in ("content", "structural")
        }
        for source in ("fresh", "correct", "wrong")
    }
    receipt = _validate_vp_receipt({
        "schema": VP_RECEIPT_SCHEMA,
        "design_id": DESIGN_ID,
        "bundle_file_sha256": file_sha256,
        "bundle_manifest_sha256": manifest_sha256,
        "descriptor_sha256": sha256_bytes(descriptor_bytes),
        "case_id": descriptor["case_id"],
        "stable_candidate_id": descriptor["stable_candidate_id"],
        "render_id": descriptor["render_id"],
        "schedule": descriptor["schedule"],
        "state_keys_sha256": _json_sha256(descriptor["state_keys"]),
        "selected_map_sha256": descriptor["selected_map_sha256"],
        "selected_token_ids_sha256": descriptor["selected_token_ids_sha256"],
        "logical_position_ids_sha256":
            descriptor["logical_position_ids_sha256"],
        "source_snapshot_sha256": {
            source: snapshot_sha256(snapshot)
            for source, snapshot in (("F", fresh), *sources.items())
        },
        "source_class_sha256": source_class_sha256,
        "status": result.status,
        "row_map": diagnostics.get("accepted_row_map"),
        "result_sha256": result_sha256,
        "result_layer_sha256": result_layer_sha256,
        "diagnostics_sha256": diagnostics_sha256,
    })
    receipt_sha256 = _json_sha256(receipt)
    return receipt, receipt_sha256, diagnostics, result


def build_placebo_receipt(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Build an outcome-blind Phase-A VP receipt from a fresh CPU reload."""

    loaded = load_verified_bundle(
        path, expected_file_sha256=expected_file_sha256,
        expected_descriptor=expected_descriptor)
    descriptor, fresh, sources = _revalidate_loaded(loaded)
    del descriptor
    receipt, receipt_sha256, diagnostics, _result = _run_bound_placebo(
        file_sha256=loaded.sha256,
        manifest_sha256=loaded.manifest_sha256,
        descriptor_bytes=loaded.descriptor_bytes,
        fresh=fresh,
        sources=sources)
    return {
        "receipt": receipt,
        "receipt_sha256": receipt_sha256,
        "diagnostics": diagnostics,
        "diagnostics_sha256": receipt["diagnostics_sha256"],
    }


def _reconstruct_arm_boundary(
    materialized: _MaterializedBundle,
    *,
    arm: str,
    placebo_receipt: Mapping[str, Any] | None,
    expected_placebo_receipt_sha256: str | None,
    require_cuda: bool,
) -> tuple[Snapshot, dict[str, Any]]:
    """Reconstruct one serial arm from a fresh clone without continuation."""

    _require(arm in PRIMARY_ARMS, "arm differs from frozen primary order")
    descriptor, fresh, sources = _revalidate_materialized(
        materialized, require_cuda=require_cuda)
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
    vp_receipt_sha256 = None
    if arm == "VP":
        _sha(expected_placebo_receipt_sha256,
             "expected externally bound VP receipt SHA")
        provided = _validate_vp_receipt(placebo_receipt or {})
        _require(_json_sha256(provided) == expected_placebo_receipt_sha256,
                 "VP receipt differs from external SHA commitment")
        recomputed, recomputed_sha, _diagnostics, result = _run_bound_placebo(
            file_sha256=materialized.file_sha256,
            manifest_sha256=materialized.manifest_sha256,
            descriptor_bytes=materialized.descriptor_bytes,
            fresh=fresh,
            sources=sources)
        _require(recomputed_sha == expected_placebo_receipt_sha256
                 and canonical_json_bytes(recomputed) == canonical_json_bytes(provided),
                 "VP receipt/result differs from deterministic recomputation")
        _require(recomputed["status"] == "AVAILABLE"
                 and result.values is not None,
                 "VP receipt is not AVAILABLE")
        device = fresh[0][0].device
        before_transfer = {
            class_name: tensor_sha256(result.values[class_name])
            for class_name in ("content", "structural")
        }
        moved_values = {
            class_name: result.values[class_name].to(
                device=device).clone().contiguous()
            for class_name in ("content", "structural")
        }
        _require(before_transfer == {
            class_name: tensor_sha256(moved_values[class_name])
            for class_name in ("content", "structural")
        } == recomputed["result_sha256"],
            "VP CPU-to-execution transfer changed result bits")
        vp = torch.cat(
            (moved_values["content"], moved_values["structural"]), dim=1)
        vp_receipt_sha256 = expected_placebo_receipt_sha256
    else:
        _require(placebo_receipt is None
                 and expected_placebo_receipt_sha256 is None,
                 "non-VP arm received a placebo receipt")
        vp = None

    output: Snapshot = []
    evidence: list[dict[str, Any]] = []
    for layer in range(REQUIRED_LAYERS):
        fresh_k, fresh_v = fresh[layer]
        after_k, after_v = fresh_k.clone(), fresh_v.clone()
        _require(after_k.data_ptr() != fresh_k.data_ptr()
                 and after_v.data_ptr() != fresh_v.data_ptr(),
                 f"arm {arm} layer {layer} did not clone fresh state")
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
        fresh_selected_k = fresh_k[..., start:end, :]
        fresh_selected_v = fresh_v[..., start:end, :]
        expected_k = (fresh_selected_k if key_source == "F"
                      else sources[key_source][layer][0])
        if value_source == "F":
            expected_v = fresh_selected_v
        elif value_source in SOURCE_HISTORIES:
            expected_v = sources[value_source][layer][1]
        else:
            assert vp is not None
            expected_v = vp[layer].permute(1, 0, 2).unsqueeze(0).contiguous()
        row = {
            "layer": layer,
            "before_k_sha256": tensor_sha256(fresh_k),
            "before_v_sha256": tensor_sha256(fresh_v),
            "after_k_sha256": tensor_sha256(after_k),
            "after_v_sha256": tensor_sha256(after_v),
            "fresh_selected_k_sha256": tensor_sha256(fresh_selected_k),
            "fresh_selected_v_sha256": tensor_sha256(fresh_selected_v),
            "source_selected_k_sha256": tensor_sha256(expected_k),
            "source_selected_v_sha256": tensor_sha256(expected_v),
            "inserted_selected_k_sha256": tensor_sha256(
                after_k[..., start:end, :]),
            "inserted_selected_v_sha256": tensor_sha256(
                after_v[..., start:end, :]),
            "outside_prefix_k_before_sha256": tensor_sha256(
                fresh_k[..., :start, :]),
            "outside_prefix_k_after_sha256": tensor_sha256(
                after_k[..., :start, :]),
            "outside_prefix_v_before_sha256": tensor_sha256(
                fresh_v[..., :start, :]),
            "outside_prefix_v_after_sha256": tensor_sha256(
                after_v[..., :start, :]),
        }
        _require(row["inserted_selected_k_sha256"] ==
                 row["source_selected_k_sha256"]
                 and row["inserted_selected_v_sha256"] ==
                 row["source_selected_v_sha256"],
                 f"arm {arm} layer {layer} selected insertion differs")
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
        "vp_receipt_sha256": vp_receipt_sha256,
        "per_layer": evidence,
        "output_sha256": snapshot_sha256(output),
    }


def reconstruct_arm_boundary(
    materialized: _MaterializedBundle,
    *,
    arm: str,
    placebo_receipt: Mapping[str, Any] | None = None,
    expected_placebo_receipt_sha256: str | None = None,
) -> tuple[Snapshot, dict[str, Any]]:
    """Public CUDA-only reconstruction from immutable external commitments."""

    return _reconstruct_arm_boundary(
        materialized,
        arm=arm,
        placebo_receipt=placebo_receipt,
        expected_placebo_receipt_sha256=expected_placebo_receipt_sha256,
        require_cuda=True)


def _reconstruct_arm_boundary_for_test(
    materialized: _MaterializedBundle,
    *,
    arm: str,
    placebo_receipt: Mapping[str, Any] | None = None,
    expected_placebo_receipt_sha256: str | None = None,
) -> tuple[Snapshot, dict[str, Any]]:
    """Private CPU unit-test wrapper; it confers no production authorization."""

    return _reconstruct_arm_boundary(
        materialized,
        arm=arm,
        placebo_receipt=placebo_receipt,
        expected_placebo_receipt_sha256=expected_placebo_receipt_sha256,
        require_cuda=False)


__all__ = [
    "MAX_BUNDLE_BYTES",
    "MAX_DESCRIPTOR_BYTES",
    "MAX_FRESH_BOUNDARY_TOKENS",
    "MAX_LIVE_CACHE_TOKENS",
    "MAX_SELECTED_ROWS",
    "SCHEMA",
    "TECHNICAL_CASE_IDS",
    "V13BundleError",
    "VP_RECEIPT_SCHEMA",
    "VerifiedBundle",
    "build_descriptor",
    "build_placebo_receipt",
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
