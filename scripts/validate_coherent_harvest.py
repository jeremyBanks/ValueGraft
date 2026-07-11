#!/usr/bin/env python3
"""Fail-closed validation before a coherent-state pod may terminate.

This validator intentionally has no imports from ``src``.  A deployment must be
validated against the frozen on-disk schema, not whatever assumptions happen to
be importable from the checkout that performs the harvest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any


SCHEMA = 2
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6"
DESIGN_ID = "coherent-state-gapped-v6"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PARAMETER_DTYPE = "torch.bfloat16"
ATTENTION_BACKEND = "eager"
MAX_TECHNICAL_LOGICAL_POSITION = 9509
SYNTHETIC_LENGTHS = (5, 64, 900, 4096, 4097, 8193)
SYNTHETIC_PARTITIONS = (
    ((5,), (2, 3)),
    ((64,), (32, 32)),
    ((900,), (32, 868)),
    ((4096,), (32, 4064)),
    ((4096, 1), (32, 4065)),
    ((4096, 4096, 1), (32, 4096, 4065)),
)
CASE_CONTINUATION_POSITIONS = {
    "c10": 8430, "c02": 8385, "c01": 8855, "c04": 8595,
    "c07": 8600, "c11": 9381, "c05": 8556, "c09": 9195,
    "c06": 8876, "c12": 9509, "c08": 8525, "c03": 8913,
}
ARMS = (
    "A_full",
    "G_fresh",
    "G_correct",
    "G_wrong",
    "G_Vcorrect",
    "G_Kcorrect",
)
FROZEN_ORDER = (
    "c10", "c02", "c01", "c04", "c07", "c11",
    "c05", "c09", "c06", "c12", "c08", "c03",
)
WRONG_DONORS = {
    "c10": "c13", "c02": "c14", "c01": "c15",
    "c04": "c16", "c07": "c17", "c11": "c18",
    "c05": "c25", "c09": "c26", "c06": "c27",
    "c12": "c28", "c08": "c29", "c03": "c30",
}
OLD_ARMS = {
    "F_fresh", "C_coherent", "W_wrong", "V_value", "K_key", "D_delta",
    "G_delta",
}
FAILURE_RE = re.compile(
    r"(?:FATAL|Traceback \(most recent call last\)|CUDA out of memory|"
    r"OutOfMemoryError|RuntimeError|CoherentStateError|WATCH_FAILURE)",
    re.IGNORECASE,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"required artifact absent: {path.name}") from exc
    except Exception as exc:
        raise ValueError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return doc


def _canonical_payload_sha256(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items()
               if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _require_payload_sha256(doc: dict[str, Any], label: str) -> None:
    if doc.get("payload_sha256") != _canonical_payload_sha256(doc):
        raise ValueError(f"{label} payload_sha256 mismatch")


def _raw_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _backend_payload_sha256(doc: dict[str, Any]) -> str:
    payload = {key: value for key, value in doc.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _sha256_ints(values: Any, label: str) -> str:
    if not isinstance(values, list) or any(
            not isinstance(value, int) for value in values):
        raise ValueError(f"{label} is not an integer array")
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _validate_backend_attestation(doc: Any, label: str) -> None:
    if not isinstance(doc, dict):
        raise ValueError(f"{label} backend attestation is not an object")
    required = {
        "requested_implementation", "model_config", "text_config",
        "text_config_is_model_config", "expected_layer_count", "layers", "sha256",
    }
    if set(doc) != required:
        raise ValueError(f"{label} backend attestation fields differ")
    if doc["requested_implementation"] != ATTENTION_BACKEND or \
            doc["expected_layer_count"] != 48:
        raise ValueError(f"{label} backend request/layer count differs")
    for scope in ("model_config", "text_config"):
        row = doc.get(scope)
        if not isinstance(row, dict) or row.get("scope") != scope:
            raise ValueError(f"{label} {scope} record malformed")
        if not isinstance(row.get("config_class"), str) or not row["config_class"]:
            raise ValueError(f"{label} {scope} class absent")
        for field in ("_attn_implementation",
                      "_attn_implementation_internal",
                      "resolved_implementation"):
            if row.get(field) != ATTENTION_BACKEND:
                raise ValueError(f"{label} {scope}.{field} is not eager")
    layers = doc.get("layers")
    if not isinstance(layers, list) or len(layers) != 48:
        raise ValueError(f"{label} backend layer coverage differs")
    for index, row in enumerate(layers):
        if not isinstance(row, dict) or row.get("layer_index") != index:
            raise ValueError(f"{label} backend layer order differs")
        for field in ("module_name", "module_class", "module_config_class"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label} layer {index} lacks {field}")
        for field in ("module_config__attn_implementation",
                      "module_config__attn_implementation_internal",
                      "resolved_implementation"):
            if row.get(field) != ATTENTION_BACKEND:
                raise ValueError(f"{label} layer {index}.{field} is not eager")
    if doc.get("sha256") != _backend_payload_sha256(doc):
        raise ValueError(f"{label} backend SHA-256 mismatch")


def _require_identity(doc: dict[str, Any], label: str) -> None:
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{label} schema is not {SCHEMA}")
    if doc.get("amendment_id") != AMENDMENT_ID:
        raise ValueError(f"{label} amendment_id mismatch")
    if doc.get("design_id") != DESIGN_ID:
        raise ValueError(f"{label} design_id mismatch")


def _require_backend_fingerprint(doc: dict[str, Any], label: str) -> None:
    if doc.get("attention_backend") != ATTENTION_BACKEND:
        raise ValueError(f"{label} does not freeze eager attention")
    _validate_backend_attestation(
        doc.get("attention_backend_fingerprint"), label)


def _validate_gate(root: Path, expected_status: str) -> dict[str, Any]:
    gate = _load(root / "production_kernel_gate.json")
    _require_identity(gate, "production gate")
    _require_payload_sha256(gate, "production gate")
    if gate.get("status") != expected_status:
        raise ValueError(
            f"production gate status {gate.get('status')!r} != {expected_status!r}")
    if not gate.get("completed_at"):
        raise ValueError("production gate lacks completed_at")
    gates = gate.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("production gate lacks gates object")
    if gate.get("model") != MODEL_ID or gate.get("revision") != MODEL_REVISION:
        raise ValueError("production gate model or revision mismatch")
    if gate.get("dtype") != PARAMETER_DTYPE:
        raise ValueError("production gate dtype mismatch")
    expected_passes = expected_status == "PASS"
    backend = gates.get("attention_backend")
    if expected_passes:
        if not isinstance(backend, dict):
            raise ValueError("production gate lacks attention-backend evidence")
        if (backend.get("observed_backend") != ATTENTION_BACKEND or
                backend.get("passes") is not True):
            raise ValueError("production gate did not attest eager attention")
        _validate_backend_attestation(
            backend.get("fingerprint"), "production gate")
    elif backend is not None:
        if not isinstance(backend, dict):
            raise ValueError("failed gate attention-backend evidence is malformed")
        if (backend.get("passes") is True and
                backend.get("observed_backend") != ATTENTION_BACKEND):
            raise ValueError("failed gate falsely attests a non-eager backend")
    if gates.get("passes") is not expected_passes:
        raise ValueError(
            f"production gates.passes must be {str(expected_passes).lower()}")
    if expected_status == "FAIL":
        evidence = (
            gate.get("error")
            or gate.get("traceback")
            or gates.get("error")
            or gates.get("traceback")
            or gates.get("failures")
        )
        if not evidence:
            raise ValueError("FAIL production gate lacks failure evidence")
    return gate


def _validate_checkpoint(doc: dict[str, Any], path: Path, *, scored: bool,
                         expected_fingerprint: dict[str, Any] | None = None) -> None:
    _require_identity(doc, path.name)
    if scored:
        if doc.get("schema") != SCHEMA or doc.get("stage") != "scored" or \
                doc.get("status") != "scored":
            raise ValueError(f"{path.name} is not a schema-2 scored checkpoint")
        required = {
            "conversation", "summary", "sources", "destination", "arm_scores",
            "conversation_outcomes", "gates", "runtime", "fingerprint",
        }
        missing = sorted(required - doc.keys())
        if missing:
            raise ValueError(f"{path.name} missing fields {missing}")
        for field in ("arm_scores", "conversation_outcomes"):
            arms = doc.get(field)
            if not isinstance(arms, dict) or set(arms) != set(ARMS):
                raise ValueError(
                    f"{path.name} {field} must contain exactly {list(ARMS)}")
            stale = OLD_ARMS.intersection(arms)
            if stale:
                raise ValueError(f"{path.name} contains retired arms {sorted(stale)}")
        gates = doc.get("gates")
        if not isinstance(gates, dict) or gates.get("technical_pass") is not True:
            raise ValueError(f"{path.name} lacks technical_pass=true")
        fingerprint = doc.get("fingerprint")
        if not isinstance(fingerprint, dict):
            raise ValueError(f"{path.name} lacks fingerprint object")
        _require_identity(fingerprint, f"{path.name} fingerprint")
        if fingerprint.get("frozen_order") != list(FROZEN_ORDER):
            raise ValueError(f"{path.name} fingerprint frozen order mismatch")
        if fingerprint.get("wrong_donors") != WRONG_DONORS:
            raise ValueError(f"{path.name} fingerprint donor map mismatch")
        _require_backend_fingerprint(fingerprint, f"{path.name} fingerprint")
        if (expected_fingerprint is not None and
                fingerprint != expected_fingerprint):
            raise ValueError(
                f"{path.name} fingerprint differs from manifest fingerprint")
    else:
        if doc.get("stage") != "void" or doc.get("status") != "void":
            raise ValueError(f"{path.name} is not a void checkpoint")
        evidence = doc.get("failure") or doc.get("error") or doc.get("failures")
        if not evidence:
            raise ValueError(f"{path.name} void checkpoint lacks failure evidence")


def _checkpoint_paths(root: Path) -> list[Path]:
    return sorted(root.glob("conv_*.json"))


def _validate_terminal_integrity(root: Path, expected_status: str) -> dict[str, Any]:
    index_path = root / "terminal_artifact_index.json"
    receipt_path = root / "terminal_receipt.json"
    index = _load(index_path)
    receipt = _load(receipt_path)
    _require_identity(index, "terminal index")
    _require_identity(receipt, "terminal receipt")
    if index.get("status") != expected_status or \
            receipt.get("status") != expected_status:
        raise ValueError("terminal envelope status mismatch")
    if receipt.get("index_path") != index_path.name:
        raise ValueError("terminal receipt index path mismatch")
    if receipt.get("index_bytes") != index_path.stat().st_size or \
            receipt.get("index_raw_sha256") != _raw_file_sha256(index_path):
        raise ValueError("terminal receipt does not match index bytes")
    canonical_gate = root / "production_kernel_gate.json"
    unique: list[Path] = []
    if canonical_gate.exists():
        unique = sorted(root.glob("production_kernel_gate_*.json"))
        if len(unique) != 1:
            raise ValueError(f"terminal run has {len(unique)} unique gate attempts")
        gate = _load(canonical_gate)
        stage_refs = ((gate.get("gates") or {}).get("stage_refs") or {})
        if not isinstance(stage_refs, dict):
            raise ValueError("technical gate stage_refs is not an object")
        expected_ref_names = {
            "committed_case_schedule_fixtures",
            "external_donor_construction",
        }
        if expected_status == "PASS" and set(stage_refs) != expected_ref_names:
            raise ValueError("PASS technical gate heavy-stage references differ")
        sidecar_paths: list[str] = []
        for stage_name, ref in sorted(stage_refs.items()):
            if stage_name not in expected_ref_names or not isinstance(ref, dict):
                raise ValueError(f"invalid technical stage reference: {stage_name}")
            relative = ref.get("path")
            expected_name = f"technical_stage_{stage_name}.json"
            if relative != expected_name:
                raise ValueError(f"technical stage path differs: {stage_name}")
            sidecar_paths.append(relative)
        required = [
            "manifest.json", "production_kernel_gate.json", unique[0].name,
            *sidecar_paths,
        ]
        if expected_status == "FAIL":
            required.append("failure.json")
        required = sorted(required)
    else:
        required = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*.json")
            if path.name not in {index_path.name, receipt_path.name})
    if index.get("required_apparatus_payload_paths") != required:
        raise ValueError("terminal index required-path set differs")
    rows = index.get("artifacts")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != required:
        raise ValueError("terminal index artifact rows differ")
    raw_hashes = {}
    for row in rows:
        path = root / row["path"]
        doc = _load(path)
        _require_identity(doc, f"indexed {row['path']}")
        _require_payload_sha256(doc, f"indexed {row['path']}")
        if row.get("bytes") != path.stat().st_size:
            raise ValueError(f"indexed byte count differs: {row['path']}")
        raw_sha = _raw_file_sha256(path)
        if row.get("raw_sha256") != raw_sha or \
                row.get("payload_sha256") != doc.get("payload_sha256"):
            raise ValueError(f"indexed hashes differ: {row['path']}")
        raw_hashes[row["path"]] = raw_sha
    if unique and canonical_gate.read_bytes() != unique[0].read_bytes():
        raise ValueError("canonical and unique gate bytes differ")
    allowed_json = set(required) | {index_path.name, receipt_path.name}
    extra = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*.json")
        if path.relative_to(root).as_posix() not in allowed_json)
    if extra:
        raise ValueError(f"terminal run has unindexed JSON payloads {extra}")
    return {
        "index_raw_sha256": _raw_file_sha256(index_path),
        "receipt_raw_sha256": _raw_file_sha256(receipt_path),
        "payload_raw_sha256": raw_hashes,
    }


def _resolve_heavy_stages(root: Path, gate: dict[str, Any]) -> dict[str, Any]:
    """Verify and materialize the two indexed heavy lifecycle sidecars."""
    gates = json.loads(json.dumps(gate.get("gates") or {}))
    refs = gates.pop("stage_refs", None)
    expected = {
        "committed_case_schedule_fixtures",
        "external_donor_construction",
    }
    if not isinstance(refs, dict) or set(refs) != expected:
        raise ValueError("technical gate heavy-stage reference set differs")
    for stage_name in sorted(expected):
        if stage_name in gates:
            raise ValueError(f"heavy stage was both inline and referenced: {stage_name}")
        ref = refs[stage_name]
        relative = f"technical_stage_{stage_name}.json"
        if not isinstance(ref, dict) or ref.get("path") != relative:
            raise ValueError(f"heavy-stage reference path differs: {stage_name}")
        path = root / relative
        sidecar = _load(path)
        _require_identity(sidecar, relative)
        _require_payload_sha256(sidecar, relative)
        if (sidecar.get("status") != gate.get("status") or
                sidecar.get("kind") != "technical_gate_stage_sidecar" or
                sidecar.get("stage_name") != stage_name or
                not isinstance(sidecar.get("lifecycle"), dict)):
            raise ValueError(f"heavy-stage sidecar identity differs: {stage_name}")
        if (ref.get("byte_count") != path.stat().st_size or
                ref.get("raw_file_sha256") != _raw_file_sha256(path) or
                ref.get("payload_sha256") != sidecar.get("payload_sha256")):
            raise ValueError(f"heavy-stage sidecar binding differs: {stage_name}")
        gates[stage_name] = sidecar["lifecycle"]
    return gates


def _find_forbidden_semantic_fields(value: Any, path: str = "") -> list[str]:
    forbidden = {
        "arm_scores", "conversation_outcomes", "calibration_outcomes",
        "technical_margins_not_semantic_outcomes", "semantic_outcomes",
        "target_scores",
    }
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in forbidden:
                found.append(child_path)
            found.extend(_find_forbidden_semantic_fields(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_semantic_fields(
                child, f"{path}[{index}]"))
    return found


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _required_pass_stage(gates: dict[str, Any], key: str) -> dict[str, Any]:
    stage = gates.get(key)
    if not isinstance(stage, dict):
        raise ValueError(f"required gate stage absent: {key}")
    if stage.get("status") != "PASS" or stage.get("passes") is not True:
        raise ValueError(f"required gate stage did not PASS: {key}")
    if stage.get("failure_evidence") not in (None, [], {}):
        raise ValueError(f"passing stage has failure evidence: {key}")
    return stage


def _validate_schedule_measurement(row: dict[str, Any], *, layers: int,
                                   tolerance: float, label: str) -> float:
    if row.get("status") != "PASS" or row.get("passes") is not True:
        raise ValueError(f"{label} schedule row did not PASS")
    if row.get("base_measurement_complete") is not True or \
            row.get("continuation_measurement_complete") is not True:
        raise ValueError(f"{label} schedule measurement is incomplete")
    if _finite(row.get("tolerance"), f"{label}.tolerance") != tolerance:
        raise ValueError(f"{label} tolerance differs")
    base = row.get("per_layer")
    continuation = row.get("continuation_per_layer")
    if not isinstance(base, list) or len(base) != layers or \
            not isinstance(continuation, list) or len(continuation) != layers:
        raise ValueError(f"{label} layer coverage differs")
    if [x.get("layer") for x in base] != list(range(layers)) or \
            [x.get("layer") for x in continuation] != list(range(layers)):
        raise ValueError(f"{label} layer order differs")
    cache_k = max(_finite(x.get("k_max_abs"), f"{label}.K") for x in base)
    cache_v = max(_finite(x.get("v_max_abs"), f"{label}.V") for x in base)
    next_k = max(_finite(x.get("k_max_abs"), f"{label}.nextK")
                 for x in continuation)
    next_v = max(_finite(x.get("v_max_abs"), f"{label}.nextV")
                 for x in continuation)
    metrics = {
        "cache_k_max_abs": cache_k,
        "cache_v_max_abs": cache_v,
        "last_logits_max_abs": _finite(
            row.get("last_logits_max_abs"), f"{label}.logits"),
        "selected_margin_abs_shift": _finite(
            row.get("selected_margin_abs_shift"), f"{label}.margin"),
        "continuation_logits_max_abs": _finite(
            row.get("continuation_logits_max_abs"), f"{label}.next_logits"),
        "continuation_k_max_abs": next_k,
        "continuation_v_max_abs": next_v,
    }
    for key, observed in metrics.items():
        if _finite(row.get(key), f"{label}.{key}") != observed:
            raise ValueError(f"{label} stored {key} aggregate differs")
    aggregate = max(metrics.values())
    if _finite(row.get("observed_aggregate"),
               f"{label}.observed_aggregate") != aggregate:
        raise ValueError(f"{label} observed aggregate differs")
    if aggregate > tolerance:
        raise ValueError(f"{label} exceeds frozen tolerance")
    return aggregate


def _validate_v6_pass_gates(gates: dict[str, Any]) -> None:
    required_order = [
        "attention_backend", "synthetic_schedule_fixtures",
        "committed_case_schedule_fixtures", "generated_replay_identity",
        "snapshot_rebuild_identity", "physical_causal_mask_identity",
        "future_mutation_identity", "position_structure",
        "intervention_propagation", "calibration_construction",
        "external_donor_construction", "retired_G_delta",
    ]
    if gates.get("stage_order") != required_order:
        raise ValueError("technical gate stage order differs")
    if gates.get("max_technical_logical_position") != \
            MAX_TECHNICAL_LOGICAL_POSITION:
        raise ValueError("technical gate maximum logical position differs")

    backend = _required_pass_stage(gates, "attention_backend")
    if backend.get("observed_coverage") != 48:
        raise ValueError("backend stage coverage differs")
    backend_raw = backend.get("raw") or {}
    _validate_backend_attestation(
        backend_raw.get("fingerprint"), "technical gate")
    subject = backend_raw.get("subject") or {}
    if (subject.get("resolved_revision") != MODEL_REVISION or
            subject.get("dtype") != PARAMETER_DTYPE or
            "cuda" not in str(subject.get("device", "")).lower() or
            int(subject.get("context_limit", -1)) <=
            MAX_TECHNICAL_LOGICAL_POSITION or
            subject.get("max_technical_logical_position") !=
            MAX_TECHNICAL_LOGICAL_POSITION or
            subject.get("context_coverage_passes") is not True):
        raise ValueError("technical subject attestation differs")

    synthetic = _required_pass_stage(gates, "synthetic_schedule_fixtures")
    if _finite(synthetic.get("threshold"), "synthetic threshold") != 5e-4:
        raise ValueError("synthetic stage threshold differs")
    raw = synthetic.get("raw") or {}
    rows = raw.get("contiguous")
    if not isinstance(rows, list) or len(rows) != len(SYNTHETIC_LENGTHS):
        raise ValueError("synthetic fixture coverage differs")
    aggregates = []
    for index, (row, length, partitions) in enumerate(zip(
            rows, SYNTHETIC_LENGTHS, SYNTHETIC_PARTITIONS)):
        if (row.get("length") != length or
                row.get("reference_partition") != list(partitions[0]) or
                row.get("alternative_partition") != list(partitions[1])):
            raise ValueError(f"synthetic fixture {index} declaration differs")
        aggregates.append(_validate_schedule_measurement(
            row, layers=48, tolerance=5e-4,
            label=f"synthetic[{length}]"))
    gap = raw.get("logical_gap")
    if not isinstance(gap, dict):
        raise ValueError("logical-gap fixture absent")
    if (gap.get("logical_positions") != list(range(32)) + list(range(8192, 8224))
            or gap.get("physical_cache_positions") != list(range(64))):
        raise ValueError("logical-gap position arrays differ")
    if gap.get("full_attention_over_physically_prior_rows") is not True:
        raise ValueError("logical-gap full-attention assertion absent")
    aggregates.append(_validate_schedule_measurement(
        gap, layers=48, tolerance=5e-4, label="logical_gap"))
    if synthetic.get("observed_coverage") != 7 or \
            _finite(synthetic.get("observed_aggregate"),
                    "synthetic aggregate") != max(aggregates):
        raise ValueError("synthetic stage aggregate/coverage differs")

    committed = _required_pass_stage(
        gates, "committed_case_schedule_fixtures")
    if _finite(committed.get("threshold"), "case threshold") != 5e-4:
        raise ValueError("case stage threshold differs")
    case_raw = committed.get("raw") or {}
    case_rows = case_raw.get("rows")
    if not isinstance(case_rows, list) or len(case_rows) != 12:
        raise ValueError("committed-case fixture coverage differs")
    case_aggregates = []
    for index, (row, cid) in enumerate(zip(case_rows, FROZEN_ORDER), 1):
        expected_tokens = CASE_CONTINUATION_POSITIONS[cid]
        if (row.get("conversation_id") != cid or
                row.get("order_position") != index or
                row.get("source_path") != f"data/synthetic/{cid}.json" or
                row.get("token_count") != expected_tokens):
            raise ValueError(f"committed-case identity differs: {cid}")
        for key in ("raw_source_file_sha256", "canonical_parsed_source_sha256",
                    "tokenizer_vocabulary_sha256", "chat_template_sha256",
                    "summary_request_sha256"):
            if not isinstance(row.get(key), str) or len(row[key]) != 64:
                raise ValueError(f"committed-case {cid} lacks {key}")
        token_ids = row.get("complete_prefix_token_ids")
        positions = row.get("complete_position_ids")
        if len(token_ids or []) != expected_tokens or \
                positions != list(range(expected_tokens)):
            raise ValueError(f"committed-case {cid} raw token/position coverage differs")
        if (_sha256_ints(token_ids, f"{cid}.tokens") !=
                row.get("complete_prefix_token_sha256") or
                _sha256_ints(positions, f"{cid}.positions") !=
                row.get("complete_position_array_sha256")):
            raise ValueError(f"committed-case {cid} token/position hash differs")
        widths = row.get("conceptual_block_widths") or {}
        if any(int(widths.get(key, 0)) <= 0
               for key in ("system", "history", "request_header")) or \
                sum(int(widths[key]) for key in widths) != expected_tokens:
            raise ValueError(f"committed-case {cid} conceptual widths differ")
        if (row.get("system_end") != widths["system"] or
                row.get("request_header_start") !=
                widths["system"] + widths["history"] or
                row.get("system_equal") is not True or
                row.get("request_header_equal") is not True):
            raise ValueError(f"committed-case {cid} boundary equality differs")
        for key in ("ordinary_resolved_call_widths",
                    "message_block_resolved_call_widths"):
            partition = row.get(key)
            if (not isinstance(partition, list) or not partition or
                    any(not isinstance(width, int) or width < 1 or width > 4096
                        for width in partition) or
                    sum(partition) != expected_tokens):
                raise ValueError(f"committed-case {cid} {key} differs")
        case_aggregates.append(_validate_schedule_measurement(
            row, layers=48, tolerance=5e-4, label=f"case[{cid}]"))
    if (committed.get("observed_coverage") != 12 or
            case_raw.get("frozen_order") != list(FROZEN_ORDER) or
            case_raw.get("coverage_exact") is not True or
            _finite(committed.get("observed_aggregate"), "case aggregate") !=
            max(case_aggregates)):
        raise ValueError("committed-case aggregate/order differs")

    generated = _required_pass_stage(gates, "generated_replay_identity")
    gen_raw = generated.get("raw") or {}
    gen_per = gen_raw.get("per_layer")
    if not isinstance(gen_per, list) or len(gen_per) != 48:
        raise ValueError("generated/replay layer coverage differs")
    gen_k = max(_finite(row.get("k_max_abs"), "generated K") for row in gen_per)
    gen_v = max(_finite(row.get("v_max_abs"), "generated V") for row in gen_per)
    gen_metrics = [
        _finite(gen_raw.get("token_logprob_max_abs"), "generated logprob"),
        gen_k, gen_v,
    ]
    if (gen_raw.get("k_max_abs") != gen_k or gen_raw.get("v_max_abs") != gen_v or
            _finite(generated.get("observed_aggregate"), "generated aggregate") !=
            max(gen_metrics) or max(gen_metrics) > 1e-4):
        raise ValueError("generated/replay aggregate differs")

    for stage_name, fields in (
        ("snapshot_rebuild_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("physical_causal_mask_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("future_mutation_identity", ("earlier_logits_max_abs", "earlier_cache_max_abs")),
    ):
        stage = _required_pass_stage(gates, stage_name)
        values = [_finite((stage.get("raw") or {}).get(field),
                          f"{stage_name}.{field}") for field in fields]
        if (stage.get("observed_coverage") != 1 or
                _finite(stage.get("observed_aggregate"),
                        f"{stage_name}.aggregate") != max(values) or
                max(values) > 1e-4):
            raise ValueError(f"{stage_name} aggregate differs")

    position = _required_pass_stage(gates, "position_structure")
    pos = position.get("raw") or {}
    for key in ("common_summary_start", "wrong_prefix_length_equal",
                "post_summary_nonempty"):
        if pos.get(key) is not True:
            raise ValueError(f"position structure lacks {key}")
    for key in ("altered_structure_failure_injection",
                "wrong_position_failure_injection"):
        if (pos.get(key) or {}).get("rejected") is not True:
            raise ValueError(f"position structure injection failed: {key}")
    physical = pos.get("physical_cache_positions")
    if physical != list(range(len(physical or []))):
        raise ValueError("physical cache positions are not contiguous")

    intervention = _required_pass_stage(gates, "intervention_propagation")
    inter = intervention.get("raw") or {}
    for key in ("fresh_self_replacement",
                "correct_insert_and_non_summary_preservation",
                "wrong_insert_and_non_summary_preservation",
                "per_arm_fork_at_summary_boundary",
                "independently_recomputed_identical_tail_lengths",
                "downstream_sensitivity", "recomputed_tail_changed"):
        if inter.get(key) is not True:
            raise ValueError(f"intervention stage lacks {key}")
    if (inter.get("pre_tailed_failure_injection") or {}).get("rejected") is not True:
        raise ValueError("pre-tailed failure injection did not reject")
    if not isinstance(inter.get("sensitivity_attempts"), list) or \
            not inter["sensitivity_attempts"]:
        raise ValueError("intervention sensitivity attempts absent")

    calibration = _required_pass_stage(gates, "calibration_construction")
    cal = calibration.get("raw") or {}
    if (cal.get("passes") is not True or cal.get("model_forwards") != 0 or
            cal.get("semantic_scoring_performed") is not False or
            calibration.get("observed_coverage") != 2):
        raise ValueError("calibration construction coverage differs")

    donors = _required_pass_stage(gates, "external_donor_construction")
    donor = donors.get("raw") or {}
    donor_rows = donor.get("rows")
    if (not isinstance(donor_rows, list) or len(donor_rows) != 12 or
            donor.get("mapping") != WRONG_DONORS or
            donor.get("frozen_order") != list(FROZEN_ORDER) or
            donor.get("n_unique_donor_ids") != 12 or
            donor.get("n_unique_donor_hashes") != 12 or
            donors.get("observed_coverage") != 12):
        raise ValueError("external donor coverage differs")
    for index, (row, cid) in enumerate(zip(donor_rows, FROZEN_ORDER), 1):
        if (row.get("order_position") != index or row.get("target_id") != cid or
                row.get("donor_id") != WRONG_DONORS[cid] or
                row.get("subject_native") is not False or
                row.get("correct_prefix_tokens") != row.get("wrong_prefix_tokens") or
                row.get("correct_wrong_length_equal") is not True or
                row.get("structural_tokens_equal") is not True or
                row.get("system_request_header_retained_tail_unchanged") is not True or
                row.get("changes_confined_to_declared_content_positions") is not True or
                row.get("replacement_spans_non_overlapping") is not True or
                row.get("replacement_coverage_exact") is not True or
                int(row.get("changed_position_count", 0)) < 1):
            raise ValueError(f"external donor row differs: {cid}")
        correct_ids = row.get("correct_prefix_ids")
        structural = row.get("structural_positions")
        content = row.get("content_positions")
        changed = row.get("changed_positions")
        for values, count_key, hash_key in (
                (correct_ids, "correct_prefix_tokens", "correct_prefix_sha256"),
                (structural, "structural_position_count",
                 "structural_positions_sha256"),
                (content, "content_position_count", "content_positions_sha256"),
                (changed, "changed_position_count", "changed_positions_sha256")):
            if (not isinstance(values, list) or row.get(count_key) != len(values) or
                    _sha256_ints(values, f"donor[{cid}].{count_key}") !=
                    row.get(hash_key)):
                raise ValueError(f"external donor row array differs: {cid}")
        if (not set(changed).issubset(content) or
                any(position < 0 or position >= len(correct_ids)
                    for position in structural + content + changed)):
            raise ValueError(f"external donor position coverage differs: {cid}")
        replacements = row.get("replacements")
        if (not isinstance(replacements, list) or
                row.get("replacement_count") != len(replacements) or
                not replacements):
            raise ValueError(f"external donor replacements differ: {cid}")
        covered: list[int] = []
        for replacement in replacements:
            if not isinstance(replacement, dict):
                raise ValueError(f"external donor replacement malformed: {cid}")
            start, end = replacement.get("start"), replacement.get("end")
            if (not isinstance(start, int) or not isinstance(end, int) or
                    not 0 <= start < end <= len(correct_ids) or
                    replacement.get("length") != end - start or
                    replacement.get("contains_special_token") is not False):
                raise ValueError(f"external donor replacement bounds differ: {cid}")
            for values_key, hash_key in (
                    ("target_ids", "target_ids_sha256"),
                    ("donor_pool_ids", "source_pool_sha256"),
                    ("replacement_ids", "replacement_sha256")):
                values = replacement.get(values_key)
                if (_sha256_ints(values, f"donor[{cid}].{values_key}") !=
                        replacement.get(hash_key)):
                    raise ValueError(f"external donor replacement hash differs: {cid}")
            if (len(replacement["target_ids"]) != end - start or
                    len(replacement["replacement_ids"]) != end - start):
                raise ValueError(f"external donor replacement length differs: {cid}")
            covered.extend(range(start, end))
        if (len(covered) != len(set(covered)) or
                sorted(covered) != sorted(content)):
            raise ValueError(f"external donor replacement coverage differs: {cid}")

    retired = _required_pass_stage(gates, "retired_G_delta")
    retired_raw = retired.get("raw") or {}
    if (retired_raw.get("executed") is not False or
            retired_raw.get("authorizes_run") is not False):
        raise ValueError("G_delta retirement assertion differs")

    if gates.get("passes") is not True or gates.get("status") != "PASS" or \
            gates.get("failures") not in ([], None):
        raise ValueError("aggregate gate verdict differs")


def _validate_complete(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "manifest")
    _require_payload_sha256(manifest, "manifest")
    if (manifest.get("status") != "COMPLETE" or
            manifest.get("phase") != "COMPLETE" or
            manifest.get("technical_only") is not False):
        raise ValueError("complete harvest lacks COMPLETE manifest")
    if manifest.get("resume_probe_verified") is not True:
        raise ValueError("manifest lacks resume_probe_verified=true")
    manifest_fingerprint = manifest.get("fingerprint")
    if not isinstance(manifest_fingerprint, dict):
        raise ValueError("manifest lacks run fingerprint")
    _require_identity(manifest_fingerprint, "manifest fingerprint")
    _require_backend_fingerprint(manifest_fingerprint, "manifest fingerprint")
    if (manifest.get("attention_backend") != ATTENTION_BACKEND or
            manifest.get("attention_backend_fingerprint") !=
            manifest_fingerprint.get("attention_backend_fingerprint")):
        raise ValueError("semantic manifest/backend fingerprint binding differs")
    checks = manifest.get("authorization_checks")
    required_checks = {
        "terminal_payloads_exact", "committed_directory_bytes_exact",
        "result_commit_is_ancestor", "result_commit_on_trunk",
        "harvest_attestation_committed_exact",
        "independent_harvest_revalidation_passed",
        "apparatus_inventory_exact", "static_data_fingerprint_exact",
        "backend_attestation_exact",
    }
    if (not isinstance(checks, dict) or set(checks) != required_checks or
            any(checks[key] is not True for key in required_checks)):
        raise ValueError("semantic authorization checks are not exact PASS")
    authorization = manifest.get("semantic_authorization")
    fingerprint_authorization = manifest_fingerprint.get(
        "semantic_authorization")
    if not isinstance(authorization, dict) or \
            not isinstance(fingerprint_authorization, dict):
        raise ValueError("semantic authorization binding is absent")
    harvest = authorization.get("harvest") or {}
    expected_binding = {
        "technical_result_commit": authorization.get("result_commit"),
        "technical_run_dir": authorization.get("run_dir"),
        "gate_payload_sha256": authorization.get("gate_payload_sha256"),
        "raw_sha256": authorization.get("raw_sha256"),
        "harvest_path": harvest.get("path"),
        "harvest_raw_sha256": harvest.get("raw_sha256"),
        "harvest_payload_sha256": harvest.get("payload_sha256"),
        "apparatus_aggregate_sha256": (
            authorization.get("apparatus_inventory") or {}).get(
                "aggregate_sha256"),
        "backend_exact": True,
        "static_fingerprint_exact": True,
    }
    if fingerprint_authorization != expected_binding:
        raise ValueError("semantic fingerprint/prior authorization binding differs")
    commit = expected_binding.get("technical_result_commit")
    if (not isinstance(commit, str) or len(commit) not in (40, 64) or
            any(char not in "0123456789abcdef" for char in commit.lower())):
        raise ValueError("semantic authorization lacks exact technical_result_commit")
    for key in (
            "gate_payload_sha256",
            "harvest_raw_sha256", "harvest_payload_sha256",
            "apparatus_aggregate_sha256"):
        value = expected_binding.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"semantic authorization lacks exact {key}")
    _validate_terminal_integrity(root, "PASS")
    probe = _load(root / "resume_probe.json")
    _require_identity(probe, "resume probe")
    _require_payload_sha256(probe, "resume probe")
    if probe.get("resume_probe_verified") is not True:
        raise ValueError("resume probe was not verified on restart")
    if (root / "production_kernel_gate.json").exists():
        raise ValueError("semantic run improperly contains a local technical gate")

    paths = _checkpoint_paths(root)
    if len(paths) not in (6, 12):
        raise ValueError(f"complete harvest has invalid checkpoint N={len(paths)}")
    if manifest.get("n_conversations") != len(paths):
        raise ValueError("semantic manifest/checkpoint count differs")
    seen_positions: list[int] = []
    seen_ids: list[str] = []
    for path in paths:
        doc = _load(path)
        _require_payload_sha256(doc, path.name)
        _validate_checkpoint(
            doc, path, scored=True,
            expected_fingerprint=manifest_fingerprint)
        position = doc.get("order_position")
        if not isinstance(position, int):
            raise ValueError(f"{path.name} lacks integer order_position")
        seen_positions.append(position)
        seen_ids.append(doc.get("conversation_id"))
    if sorted(seen_positions) != list(range(1, len(paths) + 1)):
        raise ValueError(f"checkpoint positions are not contiguous: {seen_positions}")
    if seen_ids != list(FROZEN_ORDER[:len(paths)]):
        raise ValueError(f"checkpoint frozen ID order differs: {seen_ids}")
    if "COHERENT_STATE_JOB_DONE" not in log_text:
        raise ValueError("complete harvest job log lacks completion marker")
    return {
        "status": "PASS",
        "mode": "complete",
        "n_scored": len(paths),
        "resume_probe_verified": True,
        "production_gate": "PASS",
    }


def _validate_technical(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "technical manifest")
    _require_payload_sha256(manifest, "technical manifest")
    if (manifest.get("status") != "TECHNICAL_PASS" or
            manifest.get("phase") != "TECHNICAL_COMPLETE"):
        raise ValueError("technical harvest lacks terminal TECHNICAL_PASS manifest")
    if (manifest.get("technical_only") is not True or
            manifest.get("model") != MODEL_ID or
            manifest.get("revision") != MODEL_REVISION or
            manifest.get("dtype") != PARAMETER_DTYPE or
            manifest.get("attention_backend") != ATTENTION_BACKEND):
        raise ValueError("technical manifest subject or mode differs")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("technical manifest lacks run fingerprint")
    _require_identity(fingerprint, "technical manifest fingerprint")
    _require_backend_fingerprint(fingerprint, "technical manifest fingerprint")
    integrity = _validate_terminal_integrity(root, "PASS")
    gate = _validate_gate(root, "PASS")
    if gate.get("technical_only") is not True:
        raise ValueError("technical gate is not marked technical_only")
    if (manifest.get("production_kernel_gate_payload_sha256") !=
            gate.get("payload_sha256")):
        raise ValueError("technical manifest/gate payload binding differs")
    if manifest.get("production_kernel_gate_path") != \
            "production_kernel_gate.json":
        raise ValueError("technical manifest canonical gate path differs")
    unique = manifest.get("production_kernel_gate_attempt_path")
    if (not isinstance(unique, str) or Path(unique).name != unique or
            not unique.startswith("production_kernel_gate_") or
            unique == "production_kernel_gate.json"):
        raise ValueError("technical manifest unique gate path differs")
    for field in (
            "fingerprint_static", "apparatus_inventory", "model", "revision",
            "dtype", "attention_backend", "attention_backend_fingerprint",
            "geometry", "context_limit"):
        if manifest.get(field) != gate.get(field):
            raise ValueError(f"technical manifest/gate {field} binding differs")
    expanded_gates = _resolve_heavy_stages(root, gate)
    _validate_v6_pass_gates(expanded_gates)
    forbidden = _find_forbidden_semantic_fields(gate)
    if forbidden:
        raise ValueError(
            f"technical-only gate contains semantic outcome fields {forbidden}")
    if _checkpoint_paths(root):
        raise ValueError("technical-only harvest contains conversation checkpoints")
    if "COHERENT_STATE_TECHNICAL_DONE" not in log_text:
        raise ValueError("technical harvest job log lacks completion marker")
    if "CHECKPOINT_SCORED" in log_text or "PHASE RENDER" in log_text:
        raise ValueError("technical-only log contains semantic execution markers")
    return {
        "status": "PASS",
        "mode": "technical",
        "n_scored": 0,
        "production_gate": "PASS",
        "attention_backend": ATTENTION_BACKEND,
        "raw_sha256": {
            "gate": _raw_file_sha256(root / "production_kernel_gate.json"),
            "manifest": _raw_file_sha256(root / "manifest.json"),
            "index": integrity["index_raw_sha256"],
            "receipt": integrity["receipt_raw_sha256"],
        },
    }


def _validate_failure(root: Path, log_text: str) -> dict[str, Any]:
    failure = _load(root / "failure.json") if (root / "failure.json").exists() else {}
    evidence = FAILURE_RE.search(log_text) or failure.get("error") or failure.get("traceback")
    if not evidence:
        raise ValueError("failure harvest lacks meaningful failure evidence")

    failure_fingerprint = None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _load(manifest_path)
        _require_identity(manifest, "failure manifest")
        candidate = manifest.get("fingerprint")
        if candidate is not None and not isinstance(candidate, dict):
            raise ValueError("failure manifest fingerprint is malformed")
        failure_fingerprint = candidate

    model_ready = "MODEL_READY" in log_text
    gate_path = root / "production_kernel_gate.json"
    gate_status = None
    if gate_path.exists():
        gate = _load(gate_path)
        gate_status = gate.get("status")
        if gate_status == "FAIL":
            _validate_gate(root, "FAIL")
        elif gate_status == "PASS":
            _validate_gate(root, "PASS")
        else:
            raise ValueError(f"failure harvest has incomplete gate status {gate_status!r}")
    elif model_ready:
        raise ValueError("post-MODEL_READY failure lacks complete production gate")

    paths = _checkpoint_paths(root)
    scored = 0
    voids = 0
    for path in paths:
        doc = _load(path)
        if doc.get("status") == "scored":
            if failure_fingerprint is None:
                raise ValueError(
                    "failure harvest has scored checkpoint but no manifest fingerprint")
            _validate_checkpoint(
                doc, path, scored=True,
                expected_fingerprint=failure_fingerprint)
            scored += 1
        elif doc.get("status") == "void":
            _validate_checkpoint(doc, path, scored=False)
            voids += 1
        else:
            raise ValueError(
                f"failure harvest contains non-terminal checkpoint {path.name}")

    if gate_status == "PASS" and voids < 1:
        raise ValueError("post-gate case failure lacks a void checkpoint")
    if model_ready and gate_status not in ("PASS", "FAIL"):
        raise ValueError("post-MODEL_READY failure lacks terminal gate evidence")
    return {
        "status": "PASS",
        "mode": "failure",
        "model_ready": model_ready,
        "production_gate": gate_status,
        "n_scored": scored,
        "n_void": voids,
    }


def validate(root: Path, mode: str) -> dict[str, Any]:
    log = root / "job.log"
    if not log.exists() or not log.stat().st_size:
        raise ValueError("job log missing or empty")
    log_text = log.read_text(errors="replace")
    # Parse every top-level JSON artifact before making any mode-specific claim.
    for path in root.glob("*.json"):
        _load(path)
    if mode == "complete":
        out = _validate_complete(root, log_text)
    elif mode == "technical":
        out = _validate_technical(root, log_text)
    else:
        out = _validate_failure(root, log_text)
    out["json_files"] = len(list(root.glob("*.json")))
    out["schema"] = SCHEMA
    out["amendment_id"] = AMENDMENT_ID
    out["design_id"] = DESIGN_ID
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("mode", choices=("technical", "complete", "failure"))
    output = ap.add_mutually_exclusive_group()
    output.add_argument("--read-only", action="store_true")
    output.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = validate(args.root, args.mode)
    repo = Path(__file__).resolve().parent.parent
    if args.read_only:
        sibling = Path(f"{args.root}.harvest_validation.json")
        if sibling.exists():
            attestation = _load(sibling)
            _require_identity(attestation, "harvest attestation")
            _require_payload_sha256(attestation, "harvest attestation")
            for key, value in result.items():
                if attestation.get(key) != value:
                    raise ValueError(
                        f"harvest attestation field differs: {key}")
            try:
                run_relative = args.root.resolve().relative_to(repo).as_posix()
            except ValueError as exc:
                raise ValueError("harvest run directory is outside repository") from exc
            validator = attestation.get("validator") or {}
            if (attestation.get("status") != "PASS" or
                    attestation.get("run_dir") != run_relative or
                    validator.get("path") != "scripts/validate_coherent_harvest.py" or
                    validator.get("sha256") !=
                    _raw_file_sha256(Path(__file__).resolve()) or
                    validator.get("version") != "coherent-harvest-v6"):
                raise ValueError("harvest attestation provenance differs")
    if args.output is not None:
        if args.output.exists():
            raise ValueError(f"refusing to overwrite harvest attestation: {args.output}")
        try:
            run_relative = args.root.resolve().relative_to(repo).as_posix()
        except ValueError as exc:
            raise ValueError("harvest run directory is outside repository") from exc
        attestation = {
            **result,
            "run_dir": run_relative,
            "validator": {
                "path": "scripts/validate_coherent_harvest.py",
                "sha256": _raw_file_sha256(Path(__file__).resolve()),
                "version": "coherent-harvest-v6",
            },
        }
        attestation["payload_sha256"] = _canonical_payload_sha256(attestation)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        payload = (json.dumps(
            attestation, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False) + "\n").encode()
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.output)
        if _load(args.output) != attestation:
            raise ValueError("harvest attestation read-back mismatch")
        result = attestation
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
