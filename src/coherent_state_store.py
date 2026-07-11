"""Atomic, resume-safe artifact storage for coherent-state runs."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from coherent_state_runtime import (
    AMENDMENT_ID,
    DESIGN_ID,
    GAPPED_ARM_NAMES,
)


class ArtifactError(RuntimeError):
    pass


def validate_production_backend_attestation(attestation: Any) -> None:
    """Independently validate the frozen v8 48-layer eager attestation."""
    if not isinstance(attestation, dict):
        raise ArtifactError("attention-backend fingerprint is not an object")
    expected_keys = {
        "requested_implementation", "model_config", "text_config",
        "text_config_is_model_config", "expected_layer_count", "layers", "sha256",
    }
    if set(attestation) != expected_keys:
        raise ArtifactError("attention-backend fingerprint field set differs")
    if attestation["requested_implementation"] != "eager" or \
            attestation["expected_layer_count"] != 48:
        raise ArtifactError("attention-backend request/layer count differs")
    for scope in ("model_config", "text_config"):
        row = attestation.get(scope)
        if not isinstance(row, dict) or row.get("scope") != scope:
            raise ArtifactError(f"{scope} backend record is malformed")
        if not isinstance(row.get("config_class"), str) or not row["config_class"]:
            raise ArtifactError(f"{scope} config class is absent")
        for key in ("_attn_implementation", "_attn_implementation_internal",
                    "resolved_implementation"):
            if row.get(key) != "eager":
                raise ArtifactError(f"{scope} {key} is not eager")
    layers = attestation.get("layers")
    if not isinstance(layers, list) or len(layers) != 48:
        raise ArtifactError("attention-backend layer coverage differs")
    for index, row in enumerate(layers):
        if not isinstance(row, dict) or row.get("layer_index") != index:
            raise ArtifactError("attention-backend layer order differs")
        for key in ("module_name", "module_class", "module_config_class"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ArtifactError(f"attention layer {index} lacks {key}")
        for key in ("module_config__attn_implementation",
                    "module_config__attn_implementation_internal",
                    "resolved_implementation"):
            if row.get(key) != "eager":
                raise ArtifactError(f"attention layer {index} {key} is not eager")
    payload = {key: value for key, value in attestation.items() if key != "sha256"}
    observed = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if attestation.get("sha256") != observed:
        raise ArtifactError("attention-backend fingerprint SHA-256 differs")


STAGE_RANK = {"rendered": 1, "captured": 2, "scored": 3, "void": 3}


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def checkpoint_path(run_dir: Path, order_position: int, conversation_id: str) -> Path:
    if order_position < 1 or not conversation_id:
        raise ArtifactError("invalid checkpoint identity")
    return run_dir / f"conv_{order_position:02d}_{conversation_id}.json"


def read_checkpoint(path: Path, fingerprint: dict,
                    minimum_stage: str = "rendered") -> dict | None:
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
    except Exception as exc:
        raise ArtifactError(f"unreadable checkpoint {path}: {exc}") from exc
    if doc.get("fingerprint") != fingerprint:
        raise ArtifactError(f"checkpoint fingerprint mismatch: {path}")
    stage = doc.get("stage")
    if stage not in STAGE_RANK or STAGE_RANK[stage] < STAGE_RANK[minimum_stage]:
        return None
    return doc


def save_render(path: Path, *, fingerprint: dict, order_position: int,
                conversation: dict, reply_records: list[dict]) -> dict:
    existing = read_checkpoint(path, fingerprint)
    if existing is not None:
        if existing.get("conversation") != conversation:
            raise ArtifactError(f"refusing to overwrite changed render: {path}")
        return existing
    doc = {
        "schema": 2, "amendment_id": AMENDMENT_ID, "design_id": DESIGN_ID,
        "stage": "rendered", "status": "rendered",
        "fingerprint": fingerprint,
        "order_position": order_position,
        "conversation_id": conversation.get("id"),
        "conversation": conversation,
        "reply_records": reply_records,
    }
    atomic_write_json(path, doc)
    return doc


def promote_checkpoint(path: Path, existing: dict, additions: dict,
                       stage: str) -> dict:
    if stage not in STAGE_RANK:
        raise ArtifactError(f"unknown stage: {stage}")
    current = existing.get("stage")
    if current not in STAGE_RANK or STAGE_RANK[stage] < STAGE_RANK[current]:
        raise ArtifactError(f"stage regression {current} -> {stage}")
    def merge(old, new, prefix=""):
        out = dict(old)
        for key, value in new.items():
            where = f"{prefix}.{key}" if prefix else key
            if key not in out:
                out[key] = value
            elif isinstance(out[key], dict) and isinstance(value, dict):
                out[key] = merge(out[key], value, where)
            elif out[key] != value:
                raise ArtifactError(f"promotion would overwrite field: {where}")
        return out

    out = merge(existing, additions)
    out["stage"] = stage
    out["status"] = "scored" if stage == "scored" else stage
    atomic_write_json(path, out)
    return out


def validate_scored_checkpoint(doc: dict) -> None:
    if (doc.get("schema") != 2 or doc.get("design_id") != DESIGN_ID or
            doc.get("amendment_id") != AMENDMENT_ID):
        raise ArtifactError(
            "scored checkpoint is not Amendments-1-2-3-4-5-6-7-8 schema 2")
    required = (
        "conversation", "summary", "sources", "destination", "arm_scores",
        "conversation_outcomes", "gates", "runtime",
        "pre_score_schedule_equivalence",
    )
    missing = [key for key in required if key not in doc]
    if missing:
        raise ArtifactError(f"scored checkpoint missing {missing}")
    outcomes = doc["conversation_outcomes"]
    missing_arms = [arm for arm in GAPPED_ARM_NAMES if arm not in outcomes]
    if missing_arms:
        raise ArtifactError(f"scored checkpoint missing arms {missing_arms}")
    unknown_arms = sorted(set(outcomes) - set(GAPPED_ARM_NAMES))
    if unknown_arms:
        raise ArtifactError(f"scored checkpoint has unknown arms {unknown_arms}")
    if not all(math.isfinite(float(outcomes[arm])) for arm in GAPPED_ARM_NAMES):
        raise ArtifactError("scored checkpoint has non-finite outcomes")
    if set(doc["arm_scores"]) != set(GAPPED_ARM_NAMES):
        raise ArtifactError("arm-score keys do not equal the amended G arm set")
    fingerprint = doc.get("fingerprint") or {}
    if fingerprint.get("attention_backend") != "eager":
        raise ArtifactError("checkpoint fingerprint does not freeze eager attention")
    validate_production_backend_attestation(
        fingerprint.get("attention_backend_fingerprint"))
    calibration = doc.get("calibration_outcomes") or {}
    if set(calibration) != {"G_fresh", "G_correct", "G_wrong"}:
        raise ArtifactError("calibration outcomes do not equal the amended G set")
    destination = doc.get("destination") or {}
    if destination.get("position_policy") != \
            "gapped_same_source_summary_position":
        raise ArtifactError("checkpoint lacks the amended position policy")
    if not doc["gates"].get("technical_pass"):
        raise ArtifactError("scored checkpoint claims failed technical gate")
    schedule = doc["pre_score_schedule_equivalence"]
    if (not isinstance(schedule, dict) or schedule.get("status") != "PASS" or
            schedule.get("passes") is not True or
            schedule.get("semantic_scoring_performed") is not False):
        raise ArtifactError("scored checkpoint lacks pre-score schedule PASS")
