"""Atomic, resume-safe artifact storage for coherent-state runs."""

from __future__ import annotations

import json
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
        raise ArtifactError("scored checkpoint is not Amendment-1 schema 2")
    required = (
        "conversation", "summary", "sources", "destination", "arm_scores",
        "conversation_outcomes", "gates", "runtime",
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
    calibration = doc.get("calibration_outcomes") or {}
    if set(calibration) != {"G_fresh", "G_correct", "G_wrong"}:
        raise ArtifactError("calibration outcomes do not equal the amended G set")
    destination = doc.get("destination") or {}
    if destination.get("position_policy") != \
            "gapped_same_source_summary_position":
        raise ArtifactError("checkpoint lacks the amended position policy")
    if not doc["gates"].get("technical_pass"):
        raise ArtifactError("scored checkpoint claims failed technical gate")
