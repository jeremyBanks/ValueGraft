"""Bounded, lossless tensor artifacts for coherent-state canary v12.

One artifact contains the fresh destination cache through the maximal R3
boundary plus the C/W source rows through R3 under N and P.  R1 and R2 are exact
prefixes.  This is sufficient for an independent process to reconstruct every
pre-continuation arm boundary and verify selected-channel and outside-region
lineage without retaining a full long-history cache for every arm.
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

from safetensors.torch import load_file, save_file
import torch

from coherent_canary_runtime import Snapshot, snapshot_hashes, tensor_sha256
from coherent_canary_schema import R1, R2, R3


DESIGN_ID = "coherent-state-decision-canary-v12"
SCHEMA_ID = "coherent-state-decision-canary-v12-tensor-bundle-v1"
REGIONS = (R1, R2, R3)
SOURCE_IDS = ("C_N", "W_N", "C_P", "W_P")
ARM_SOURCES = {
    "FF": ("F", "F"), "FC": ("F", "C"), "FW": ("F", "W"),
    "CF": ("C", "F"), "CC": ("C", "C"), "CW": ("C", "W"),
    "WF": ("W", "F"), "WC": ("W", "C"), "WW": ("W", "W"),
}
MAX_LAYERS = 48
MAX_KV_HEADS = 8
MAX_HEAD_DIM = 256
MAX_FRESH_R3_TOKENS = 512
MAX_SELECTED_R3_ROWS = 256
MAX_BUNDLE_BYTES = 256 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class CanaryArtifactError(RuntimeError):
    """A lossless tensor artifact violates its frozen bounds or commitments."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryArtifactError(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _intervals(value: Any, label: str) -> dict[str, list[int]]:
    _require(isinstance(value, Mapping) and set(value) == set(REGIONS),
             f"{label} region set differs")
    result: dict[str, list[int]] = {}
    common_start = None
    previous_end = None
    for region in REGIONS:
        interval = value[region]
        _require(isinstance(interval, (list, tuple)) and len(interval) == 2 and
                 all(isinstance(item, int) for item in interval),
                 f"{label} {region} interval is invalid")
        start, end = int(interval[0]), int(interval[1])
        _require(0 <= start < end, f"{label} {region} interval is empty")
        if common_start is None:
            common_start = start
        _require(start == common_start, f"{label} regions do not share R1 start")
        if previous_end is not None:
            _require(end > previous_end, f"{label} region ends are not increasing")
        previous_end = end
        result[region] = [start, end]
    return result


def validate_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema", "design_id", "case_id", "fingerprint_sha256",
        "fresh_plan_sha256", "source_plan_sha256s",
        "source_capture_bindings", "cache_geometry",
        "fresh_physical_regions", "source_logical_regions",
    }
    _require(isinstance(descriptor, Mapping) and set(descriptor) == expected,
             "tensor descriptor field set differs")
    _require(descriptor["schema"] == SCHEMA_ID and
             descriptor["design_id"] == DESIGN_ID,
             "tensor descriptor schema/design differs")
    _require(isinstance(descriptor["case_id"], str) and descriptor["case_id"],
             "tensor descriptor case is empty")
    for key in ("fingerprint_sha256", "fresh_plan_sha256"):
        _require(isinstance(descriptor[key], str) and
                 SHA256_RE.fullmatch(descriptor[key]) is not None,
                 f"tensor descriptor {key} is not SHA-256")
    plans = descriptor["source_plan_sha256s"]
    _require(isinstance(plans, Mapping) and set(plans) == set(SOURCE_IDS) and
             all(isinstance(value, str) and SHA256_RE.fullmatch(value)
                 for value in plans.values()),
             "tensor descriptor source-plan bindings differ")
    geometry = descriptor["cache_geometry"]
    _require(isinstance(geometry, Mapping) and set(geometry) == {
        "layers", "batch", "kv_heads", "head_dim", "dtype"},
        "tensor descriptor cache geometry fields differ")
    _require(isinstance(geometry["layers"], int) and
             1 <= geometry["layers"] <= MAX_LAYERS and
             geometry["batch"] == 1 and
             isinstance(geometry["kv_heads"], int) and
             1 <= geometry["kv_heads"] <= MAX_KV_HEADS and
             isinstance(geometry["head_dim"], int) and
             1 <= geometry["head_dim"] <= MAX_HEAD_DIM and
             geometry["dtype"] == "torch.bfloat16",
             "tensor descriptor cache geometry exceeds frozen bounds")
    fresh = _intervals(descriptor["fresh_physical_regions"], "fresh physical")
    source_raw = descriptor["source_logical_regions"]
    _require(isinstance(source_raw, Mapping) and set(source_raw) == {"N", "P"},
             "source logical schedules differ")
    source = {schedule: _intervals(source_raw[schedule],
                                   f"source {schedule} logical")
              for schedule in ("N", "P")}
    _require(source["N"] == source["P"],
             "source N/P logical region intervals differ")
    for region in REGIONS:
        widths = {fresh[region][1] - fresh[region][0]}
        widths.update(source[schedule][region][1] -
                      source[schedule][region][0] for schedule in ("N", "P"))
        _require(len(widths) == 1,
                 f"fresh/N/P {region} selected widths differ")
    captures = descriptor["source_capture_bindings"]
    _require(isinstance(captures, Mapping) and set(captures) == set(SOURCE_IDS),
             "source capture binding set differs")
    frozen_captures = {}
    token_hashes = set()
    for source_id in SOURCE_IDS:
        row = captures[source_id]
        _require(isinstance(row, Mapping) and set(row) == {
            "plan_sha256", "history", "schedule", "logical_start",
            "logical_end", "selected_token_ids_sha256",
            "source_execution_snapshot_sha256", "extraction_call_trace_sha256",
        }, f"{source_id} capture binding fields differ")
        history, schedule = source_id.split("_")
        expected_interval = source[schedule][R3]
        _require(row["plan_sha256"] == plans[source_id] and
                 row["history"] == history and row["schedule"] == schedule and
                 row["logical_start"] == expected_interval[0] and
                 row["logical_end"] == expected_interval[1],
                 f"{source_id} capture origin differs")
        for key in ("selected_token_ids_sha256",
                    "source_execution_snapshot_sha256",
                    "extraction_call_trace_sha256"):
            _require(isinstance(row[key], str) and SHA256_RE.fullmatch(row[key]),
                     f"{source_id} capture {key} is not SHA-256")
        token_hashes.add(row["selected_token_ids_sha256"])
        frozen_captures[source_id] = dict(row)
    _require(len(token_hashes) == 1,
             "C/W/N/P selected carrier token hashes differ")
    result = deepcopy(dict(descriptor))
    result["fresh_physical_regions"] = fresh
    result["source_logical_regions"] = source
    result["source_plan_sha256s"] = dict(plans)
    result["source_capture_bindings"] = frozen_captures
    result["cache_geometry"] = dict(geometry)
    return result


def _snapshot_geometry(snapshot: Snapshot, label: str, *, max_tokens: int) -> dict:
    _require(isinstance(snapshot, list) and 1 <= len(snapshot) <= MAX_LAYERS,
             f"{label} layer count lies outside 1..{MAX_LAYERS}")
    reference = None
    for layer, pair in enumerate(snapshot):
        _require(isinstance(pair, tuple) and len(pair) == 2,
                 f"{label} layer {layer} is not a K/V pair")
        keys, values = pair
        _require(isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor),
                 f"{label} layer {layer} lacks tensors")
        _require(keys.device.type == values.device.type == "cpu",
                 f"{label} layer {layer} is not evicted to CPU")
        _require(keys.dtype == values.dtype == torch.bfloat16,
                 f"{label} layer {layer} is not bf16")
        _require(torch.isfinite(keys).all().item() and
                 torch.isfinite(values).all().item(),
                 f"{label} layer {layer} contains nonfinite state")
        _require(keys.ndim == values.ndim == 4 and keys.shape == values.shape,
                 f"{label} layer {layer} K/V geometry differs")
        batch, heads, tokens, head_dim = map(int, keys.shape)
        _require(batch == 1 and 1 <= heads <= MAX_KV_HEADS and
                 1 <= head_dim <= MAX_HEAD_DIM and
                 1 <= tokens <= max_tokens,
                 f"{label} layer {layer} geometry exceeds bounds")
        geometry = {"batch": batch, "heads": heads, "tokens": tokens,
                    "head_dim": head_dim, "dtype": str(keys.dtype)}
        if reference is None:
            reference = geometry
        _require(geometry == reference, f"{label} layer geometry is inconsistent")
    assert reference is not None
    return {"layers": len(snapshot), **reference}


def _validate_snapshots(fresh_r3: Snapshot,
                        sources: Mapping[str, Snapshot],
                        descriptor: Mapping[str, Any]) -> dict[str, Any]:
    frozen = validate_descriptor(descriptor)
    _require(isinstance(sources, Mapping) and set(sources) == set(SOURCE_IDS),
             "tensor source set differs")
    fresh_geometry = _snapshot_geometry(
        fresh_r3, "fresh R3", max_tokens=MAX_FRESH_R3_TOKENS)
    declared_geometry = frozen["cache_geometry"]
    for observed, declared in (
        (fresh_geometry["layers"], declared_geometry["layers"]),
        (fresh_geometry["batch"], declared_geometry["batch"]),
        (fresh_geometry["heads"], declared_geometry["kv_heads"]),
        (fresh_geometry["head_dim"], declared_geometry["head_dim"]),
        (fresh_geometry["dtype"], declared_geometry["dtype"]),
    ):
        _require(observed == declared,
                 "snapshot cache geometry differs from descriptor")
    fresh_regions = frozen["fresh_physical_regions"]
    _require(fresh_geometry["tokens"] == fresh_regions[R3][1],
             "fresh snapshot does not end at R3 boundary")
    selected_width = fresh_regions[R3][1] - fresh_regions[R3][0]
    _require(selected_width <= MAX_SELECTED_R3_ROWS,
             "selected R3 rows exceed frozen bound")
    for source_id in SOURCE_IDS:
        geometry = _snapshot_geometry(
            sources[source_id], source_id, max_tokens=MAX_SELECTED_R3_ROWS)
        _require(geometry["tokens"] == selected_width,
                 f"{source_id} does not contain content-start through R3")
        for key in ("layers", "batch", "heads", "head_dim", "dtype"):
            _require(geometry[key] == fresh_geometry[key],
                     f"{source_id} {key} differs from fresh")
    return frozen


def _tensor_key(source_id: str, layer: int, channel: str) -> str:
    return f"{source_id}.layer_{layer:03d}.{channel}"


def _tensor_index(tensors: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [{
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_sha256(tensor),
    } for name, tensor in sorted(tensors.items())]


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor],
                        metadata: Mapping[str, str]) -> None:
    _require(not path.exists(), f"refusing to overwrite tensor artifact: {path}")
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
            raise CanaryArtifactError(
                f"tensor artifact appeared during write: {path}") from exc
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


def save_tensor_bundle(path: Path, *, fresh_r3: Snapshot,
                       sources: Mapping[str, Snapshot],
                       descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one bounded case bundle without replacing an existing file."""
    frozen = _validate_snapshots(fresh_r3, sources, descriptor)
    tensors: dict[str, torch.Tensor] = {}
    for source_id, snapshot in (("fresh", fresh_r3), *sources.items()):
        for layer, (keys, values) in enumerate(snapshot):
            tensors[_tensor_key(source_id, layer, "k")] = \
                keys.detach().contiguous().cpu()
            tensors[_tensor_key(source_id, layer, "v")] = \
                values.detach().contiguous().cpu()
    index = _tensor_index(tensors)
    payload_bytes = sum(tensor.numel() * tensor.element_size()
                        for tensor in tensors.values())
    _require(payload_bytes <= MAX_BUNDLE_BYTES - (1 << 20),
             "tensor bundle payload exceeds frozen byte bound")
    manifest = {"descriptor": frozen, "tensor_index": index}
    manifest["tensor_index_sha256"] = hashlib.sha256(
        _canonical(index).encode("utf-8")).hexdigest()
    _atomic_safetensors(path, tensors, {
        "schema": SCHEMA_ID,
        "manifest": _canonical(manifest),
    })
    _require(path.stat().st_size <= MAX_BUNDLE_BYTES,
             "serialized tensor bundle exceeds frozen byte bound")
    return {
        "path": str(path), "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size, "manifest": manifest,
    }


def _load_tensor_bundle(path: Path, *, expected_file_sha256: str | None,
                        expected_descriptor: Mapping[str, Any] | None,
                        verified: bool) -> dict[str, Any]:
    _require(path.is_file(), f"tensor artifact is absent: {path}")
    observed_file_hash = file_sha256(path)
    if expected_file_sha256 is not None:
        _require(observed_file_hash == expected_file_sha256,
                 "tensor artifact file hash differs")
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    _require(metadata.get("schema") == SCHEMA_ID and
             isinstance(metadata.get("manifest"), str),
             "tensor artifact metadata differs")
    try:
        manifest = json.loads(metadata["manifest"])
    except Exception as exc:
        raise CanaryArtifactError(f"tensor manifest is invalid JSON: {exc}") from exc
    _require(isinstance(manifest, Mapping) and set(manifest) == {
        "descriptor", "tensor_index", "tensor_index_sha256"},
        "tensor manifest field set differs")
    descriptor = validate_descriptor(manifest["descriptor"])
    if expected_descriptor is not None:
        _require(descriptor == validate_descriptor(expected_descriptor),
                 "tensor descriptor differs from expected")
    tensors = load_file(str(path), device="cpu")
    index = _tensor_index(tensors)
    _require(index == manifest["tensor_index"] and
             manifest["tensor_index_sha256"] == hashlib.sha256(
                 _canonical(index).encode("utf-8")).hexdigest(),
             "tensor index/hash differs")
    layers = sum(1 for row in index if row["name"].startswith("fresh.") and
                 row["name"].endswith(".k"))
    expected_names = {
        _tensor_key(source_id, layer, channel)
        for source_id in ("fresh", *SOURCE_IDS)
        for layer in range(layers) for channel in ("k", "v")
    }
    _require(set(tensors) == expected_names,
             "tensor artifact key/layer coverage differs")
    fresh = snapshot_from_loaded(tensors, "fresh", layers)
    sources = {source_id: snapshot_from_loaded(tensors, source_id, layers)
               for source_id in SOURCE_IDS}
    _validate_snapshots(fresh, sources, descriptor)
    return {
        "path": str(path), "sha256": observed_file_hash,
        "manifest": deepcopy(dict(manifest)), "tensors": tensors,
        "fresh": fresh, "sources": sources, "verified": verified,
    }


def inspect_tensor_bundle(path: Path) -> dict[str, Any]:
    """Inspect a self-consistent file without treating it as externally bound."""
    return _load_tensor_bundle(
        path, expected_file_sha256=None, expected_descriptor=None,
        verified=False)


def load_tensor_bundle(path: Path, *, expected_file_sha256: str,
                       expected_descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Load only when both independent outer commitments are supplied."""
    _require(isinstance(expected_file_sha256, str) and
             SHA256_RE.fullmatch(expected_file_sha256) is not None,
             "verified tensor load requires an external file SHA-256")
    return _load_tensor_bundle(
        path, expected_file_sha256=expected_file_sha256,
        expected_descriptor=expected_descriptor, verified=True)


def snapshot_from_loaded(tensors: Mapping[str, torch.Tensor], source_id: str,
                         layers: int) -> Snapshot:
    return [(tensors[_tensor_key(source_id, layer, "k")].clone(),
             tensors[_tensor_key(source_id, layer, "v")].clone())
            for layer in range(layers)]


def _selected_rows(loaded: Mapping[str, Any], source: str, schedule: str,
                   width: int, *, channel: int,
                   fresh: Snapshot, sources: Mapping[str, Snapshot]
                   ) -> list[torch.Tensor]:
    if source == "F":
        start = loaded["manifest"]["descriptor"]["fresh_physical_regions"][R1][0]
        return [pair[channel][..., start:start + width, :].clone()
                for pair in fresh]
    source_id = f"{source}_{schedule}"
    _require(source_id in SOURCE_IDS, f"unknown selected-row source {source_id}")
    return [pair[channel][..., :width, :].clone()
            for pair in sources[source_id]]


def _revalidate_loaded(loaded: Mapping[str, Any]) -> tuple[Snapshot,
                                                            dict[str, Snapshot]]:
    _require(isinstance(loaded, Mapping) and isinstance(loaded.get("manifest"), Mapping)
             and isinstance(loaded.get("tensors"), Mapping) and
             loaded.get("verified") is True,
             "loaded tensor bundle structure differs")
    tensors = loaded["tensors"]
    index = _tensor_index(tensors)
    _require(index == loaded["manifest"].get("tensor_index"),
             "loaded tensor payload differs from bound index")
    layers = sum(1 for row in index if row["name"].startswith("fresh.") and
                 row["name"].endswith(".k"))
    fresh = snapshot_from_loaded(tensors, "fresh", layers)
    sources = {source_id: snapshot_from_loaded(tensors, source_id, layers)
               for source_id in SOURCE_IDS}
    _validate_snapshots(fresh, sources, loaded["manifest"]["descriptor"])
    return fresh, sources


def reconstruct_arm_boundary(loaded: Mapping[str, Any], *, region: str,
                             schedule: str, cell: str) -> tuple[Snapshot, dict]:
    """Independently reconstruct one arm immediately after row replacement."""
    _require(region in REGIONS and schedule in ("N", "P") and cell in ARM_SOURCES,
             "arm reconstruction selector differs")
    regions = loaded["manifest"]["descriptor"]["fresh_physical_regions"]
    fresh, sources = _revalidate_loaded(loaded)
    start, end = regions[region]
    width = end - start
    key_source, value_source = ARM_SOURCES[cell]
    selected_k = _selected_rows(
        loaded, key_source, schedule, width, channel=0,
        fresh=fresh, sources=sources)
    selected_v = _selected_rows(
        loaded, value_source, schedule, width, channel=1,
        fresh=fresh, sources=sources)
    output: Snapshot = []
    evidence = []
    for layer, ((fresh_k_all, fresh_v_all), source_k, source_v) in enumerate(
            zip(fresh, selected_k, selected_v)):
        fresh_k = fresh_k_all[..., :end, :].clone()
        fresh_v = fresh_v_all[..., :end, :].clone()
        after_k, after_v = fresh_k.clone(), fresh_v.clone()
        after_k[..., start:end, :] = source_k
        after_v[..., start:end, :] = source_v
        output.append((after_k, after_v))
        evidence.append({
            "layer": layer,
            "fresh_before_k_sha256": tensor_sha256(fresh_k),
            "fresh_before_v_sha256": tensor_sha256(fresh_v),
            "source_selected_k_sha256": tensor_sha256(source_k),
            "source_selected_v_sha256": tensor_sha256(source_v),
            "fresh_selected_k_sha256": tensor_sha256(
                fresh_k[..., start:end, :]),
            "fresh_selected_v_sha256": tensor_sha256(
                fresh_v[..., start:end, :]),
            "inserted_selected_k_sha256": tensor_sha256(
                after_k[..., start:end, :]),
            "inserted_selected_v_sha256": tensor_sha256(
                after_v[..., start:end, :]),
            "outside_prefix_k_before_sha256": tensor_sha256(
                fresh_k[..., :start, :]),
            "outside_prefix_v_before_sha256": tensor_sha256(
                fresh_v[..., :start, :]),
            "outside_prefix_k_after_sha256": tensor_sha256(
                after_k[..., :start, :]),
            "outside_prefix_v_after_sha256": tensor_sha256(
                after_v[..., :start, :]),
            "after_k_sha256": tensor_sha256(after_k),
            "after_v_sha256": tensor_sha256(after_v),
        })
        row = evidence[-1]
        _require(row["inserted_selected_k_sha256"] ==
                 row["source_selected_k_sha256"] and
                 row["inserted_selected_v_sha256"] ==
                 row["source_selected_v_sha256"],
                 f"layer {layer} selected insertion differs")
        _require(row["outside_prefix_k_before_sha256"] ==
                 row["outside_prefix_k_after_sha256"] and
                 row["outside_prefix_v_before_sha256"] ==
                 row["outside_prefix_v_after_sha256"],
                 f"layer {layer} outside prefix changed")
        if key_source == "F":
            _require(row["inserted_selected_k_sha256"] ==
                     row["fresh_selected_k_sha256"],
                     f"layer {layer} F key source is not fresh")
        if value_source == "F":
            _require(row["inserted_selected_v_sha256"] ==
                     row["fresh_selected_v_sha256"],
                     f"layer {layer} F value source is not fresh")
    return output, {
        "region": region, "schedule": schedule, "cell": cell,
        "destination_start": start, "destination_end": end,
        "row_count": width, "key_source": key_source,
        "value_source": value_source,
        "per_layer": evidence,
        "after_snapshot_hashes": snapshot_hashes(output),
    }
