#!/usr/bin/env python3
"""Fail-closed validation before a coherent-state pod may terminate.

This validator intentionally has no imports from ``src``.  A deployment must be
validated against the frozen on-disk schema, not whatever assumptions happen to
be importable from the checkout that performs the harvest.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = 2
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4"
DESIGN_ID = "coherent-state-gapped-v4"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PARAMETER_DTYPE = "torch.bfloat16"
ATTENTION_BACKEND = "eager"
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
    attestation = doc.get("attention_backend_fingerprint")
    if not isinstance(attestation, dict) or not attestation:
        raise ValueError(f"{label} lacks attention-backend fingerprint")
    if attestation.get("requested_implementation") != ATTENTION_BACKEND:
        raise ValueError(f"{label} backend request is not eager")
    layers = attestation.get("layers")
    if not isinstance(layers, list) or len(layers) != 48:
        raise ValueError(f"{label} backend fingerprint does not cover 48 layers")
    if any(row.get("resolved_implementation") != ATTENTION_BACKEND
           for row in layers if isinstance(row, dict)) or \
            any(not isinstance(row, dict) for row in layers):
        raise ValueError(f"{label} contains a non-eager attention layer")


def _validate_gate(root: Path, expected_status: str) -> dict[str, Any]:
    gate = _load(root / "production_kernel_gate.json")
    _require_identity(gate, "production gate")
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
    backend = gates.get("attention_backend")
    if not isinstance(backend, dict):
        raise ValueError("production gate lacks attention-backend evidence")
    if (backend.get("observed_backend") != ATTENTION_BACKEND or
            backend.get("passes") is not True):
        raise ValueError("production gate did not attest eager attention")
    fingerprint = backend.get("fingerprint")
    if not isinstance(fingerprint, dict) or \
            fingerprint.get("requested_implementation") != ATTENTION_BACKEND:
        raise ValueError("production gate backend fingerprint is malformed")
    expected_passes = expected_status == "PASS"
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


def _validate_complete(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "manifest")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("complete harvest lacks COMPLETE manifest")
    if manifest.get("resume_probe_verified") is not True:
        raise ValueError("manifest lacks resume_probe_verified=true")
    manifest_fingerprint = manifest.get("fingerprint")
    if not isinstance(manifest_fingerprint, dict):
        raise ValueError("manifest lacks run fingerprint")
    _require_identity(manifest_fingerprint, "manifest fingerprint")
    _require_backend_fingerprint(manifest_fingerprint, "manifest fingerprint")
    probe = _load(root / "resume_probe.json")
    _require_identity(probe, "resume probe")
    if probe.get("resume_probe_verified") is not True:
        raise ValueError("resume probe was not verified on restart")
    _validate_gate(root, "PASS")

    paths = _checkpoint_paths(root)
    if len(paths) not in (6, 12):
        raise ValueError(f"complete harvest has invalid checkpoint N={len(paths)}")
    seen_positions: list[int] = []
    seen_ids: list[str] = []
    for path in paths:
        doc = _load(path)
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
        "mode": "complete",
        "n_scored": len(paths),
        "resume_probe_verified": True,
        "production_gate": "PASS",
    }


def _validate_technical(root: Path, log_text: str) -> dict[str, Any]:
    manifest = _load(root / "manifest.json")
    _require_identity(manifest, "technical manifest")
    if (manifest.get("status") != "TECHNICAL_PASS" or
            manifest.get("phase") != "TECHNICAL_COMPLETE"):
        raise ValueError("technical harvest lacks terminal TECHNICAL_PASS manifest")
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("technical manifest lacks run fingerprint")
    _require_identity(fingerprint, "technical manifest fingerprint")
    _require_backend_fingerprint(fingerprint, "technical manifest fingerprint")
    gate = _validate_gate(root, "PASS")
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
        "mode": "technical",
        "n_scored": 0,
        "production_gate": "PASS",
        "attention_backend": ATTENTION_BACKEND,
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
    args = ap.parse_args()
    print(json.dumps(validate(args.root, args.mode), sort_keys=True))


if __name__ == "__main__":
    main()
