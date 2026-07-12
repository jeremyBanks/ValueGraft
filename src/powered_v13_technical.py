"""Technical-only e01/long planning for the powered-v13 Stage-T canary.

This module has no production-pool, recipe, permutation, carrier-generation,
probe, score, or analysis import.  It exposes only two hash-allowlisted
engineering inputs: the historical e01 bytes and a deterministic long-geometry
extension of those same C/W histories.  Both use one literal neutral carrier;
neither is a semantic population unit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from powered_v13_schema import DESIGN_ID, N_SCHEDULE, R2
from powered_v13_tokens import (
    build_fresh_destination_plan,
    build_role_native_plan,
)


TECHNICAL_INPUT_SCHEMA = "coherent-state-powered-successor-v13-technical-input-v1"
TECHNICAL_CASE_IDS = ("technical_e01", "technical_long")
E01_RELATIVE_PATH = "data/coherent_canary_v12/revision2/session_d/e01.json"
E01_SHA256 = "6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090"
TECHNICAL_CARRIER = (
    "The earlier exchange established shared operating context and confirmed "
    "that decision criteria were recorded. This handoff preserves continuity "
    "for the next assistant while withholding the criteria themselves and "
    "every settled detail. Continue from the visible conversation, using the "
    "available context for any later request."
)
TECHNICAL_LONG_FILLER_UNIT = (
    " Shared technical context remains available for geometry verification only."
)
TECHNICAL_LONG_FILLER_REPEATS = 325
TECHNICAL_LONG_ASSISTANT = "Noted."
TECHNICAL_LONG_SOURCE_TOKEN_BOUNDS = (4500, 4600)
MAX_TECHNICAL_SOURCE_TOKENS = 7000


class V13TechnicalError(RuntimeError):
    """A technical-only input or plan differs from the Stage-T contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V13TechnicalError(message)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V13TechnicalError(f"value is not canonical-JSON safe: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"technical e01 JSON duplicates key {key!r}")
        result[key] = value
    return result


def _messages(value: object, label: str) -> list[dict[str, str]]:
    _require(isinstance(value, list) and len(value) == 17,
             f"{label} history width differs")
    rows: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, Mapping) and set(raw) == {"role", "content"},
                 f"{label} message {index} fields differ")
        role = raw.get("role")
        content = raw.get("content")
        expected_role = "system" if index == 0 else (
            "user" if index % 2 else "assistant")
        _require(role == expected_role,
                 f"{label} message {index} role differs")
        _require(isinstance(content, str) and bool(content.strip())
                 and "<|im_" not in content,
                 f"{label} message {index} content is invalid")
        rows.append({"role": role, "content": content})
    return rows


def load_fixed_e01(repo: Path) -> dict[str, Any]:
    """Read the sole allowed semantic-shaped Stage-T input by exact bytes."""

    root = Path(repo).resolve()
    path = root / E01_RELATIVE_PATH
    _require(not path.is_symlink() and path.is_file(),
             "technical e01 is absent or symlinked")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise V13TechnicalError("technical e01 escapes repository root") from exc
    raw = path.read_bytes()
    _require(sha256_bytes(raw) == E01_SHA256,
             "technical e01 byte hash differs")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V13TechnicalError(f"technical e01 is not strict UTF-8 JSON: {exc}") from exc
    _require(isinstance(value, Mapping)
             and value.get("design_id") == "coherent-state-decision-canary-v12"
             and value.get("case_id") == "e01"
             and value.get("middle_end_msg") == 15,
             "technical e01 identity differs")
    variants = value.get("variants")
    _require(isinstance(variants, Mapping)
             and set(variants) == {"correct", "wrong_focal"},
             "technical e01 variant set differs")
    correct_raw = variants["correct"]
    wrong_raw = variants["wrong_focal"]
    _require(isinstance(correct_raw, Mapping)
             and set(correct_raw) == {"messages"}
             and isinstance(wrong_raw, Mapping)
             and set(wrong_raw) == {"messages"},
             "technical e01 variant fields differ")
    correct = _messages(correct_raw["messages"], "technical e01 C")
    wrong = _messages(wrong_raw["messages"], "technical e01 W")
    middle = 15
    _require(correct[0] == wrong[0], "technical e01 system messages differ")
    _require(correct[middle:] == wrong[middle:],
             "technical e01 retained tails differ")
    _require(correct != wrong, "technical e01 C/W histories are identical")
    return {
        "source_path": E01_RELATIVE_PATH,
        "source_sha256": E01_SHA256,
        "correct": correct,
        "wrong": wrong,
        "middle_end_msg": middle,
    }


def _long_history(history: list[dict[str, str]], middle: int) -> tuple[
        list[dict[str, str]], int]:
    filler = (TECHNICAL_LONG_FILLER_UNIT
              * TECHNICAL_LONG_FILLER_REPEATS).strip()
    extended = deepcopy(history[:middle]) + [
        {"role": "user", "content": filler},
        {"role": "assistant", "content": TECHNICAL_LONG_ASSISTANT},
    ] + deepcopy(history[middle:])
    return extended, middle + 2


def _case_material(e01: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    _require(case_id in TECHNICAL_CASE_IDS, "unknown technical case ID")
    correct = deepcopy(e01["correct"])
    wrong = deepcopy(e01["wrong"])
    middle = int(e01["middle_end_msg"])
    if case_id == "technical_long":
        correct, correct_middle = _long_history(correct, middle)
        wrong, wrong_middle = _long_history(wrong, middle)
        _require(correct_middle == wrong_middle, "technical long middles differ")
        middle = correct_middle
    stable_id = sha256_json([
        DESIGN_ID,
        case_id,
        E01_SHA256,
        sha256_bytes(TECHNICAL_CARRIER.encode("utf-8")),
        TECHNICAL_LONG_FILLER_UNIT if case_id == "technical_long" else None,
        TECHNICAL_LONG_FILLER_REPEATS if case_id == "technical_long" else None,
    ])
    return {
        "schema": TECHNICAL_INPUT_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": case_id,
        "stable_technical_id": stable_id,
        "semantic_n": 0,
        "schedule": N_SCHEDULE,
        "source_path": E01_RELATIVE_PATH,
        "source_sha256": E01_SHA256,
        "carrier": TECHNICAL_CARRIER,
        "correct": correct,
        "wrong": wrong,
        "middle_end_msg": middle,
    }


def _plan_evidence(plan, *, destination: bool) -> dict[str, Any]:
    geometry = plan.geometry()
    if destination:
        start, end = plan.physical_regions.interval(R2)
        logical_positions = list(plan.logical_positions[start:end])
        selected_ids = list(plan.token_ids[start:end])
    else:
        start, end = plan.regions.interval(R2)
        logical_positions = list(range(start, end))
        selected_ids = list(plan.token_ids[start:end])
    return {
        "token_count": len(plan.token_ids),
        "token_ids_sha256": sha256_json(list(plan.token_ids)),
        "geometry_sha256": sha256_json(geometry),
        "event_count": len(plan.events),
        "max_call_width": max(event.width for event in plan.events),
        "r2_start": start,
        "r2_end": end,
        "r2_width": end - start,
        "r2_token_ids": selected_ids,
        "r2_token_ids_sha256": sha256_json(selected_ids),
        "r2_logical_positions": logical_positions,
        "r2_logical_positions_sha256": sha256_json(logical_positions),
    }


def build_technical_case(tokenizer, repo: Path, case_id: str) -> dict[str, Any]:
    """Build and mechanically prove one of the two semantic-N=0 plan sets."""

    material = _case_material(load_fixed_e01(repo), case_id)
    middle = material["middle_end_msg"]
    carrier = material["carrier"]
    correct_plan = build_role_native_plan(
        tokenizer,
        material["correct"],
        middle_end_msg=middle,
        carrier_content=carrier,
    )
    wrong_plan = build_role_native_plan(
        tokenizer,
        material["wrong"],
        middle_end_msg=middle,
        carrier_content=carrier,
    )
    fresh_correct = build_fresh_destination_plan(
        tokenizer,
        material["correct"],
        middle_end_msg=middle,
        carrier_content=carrier,
    )
    fresh_wrong = build_fresh_destination_plan(
        tokenizer,
        material["wrong"],
        middle_end_msg=middle,
        carrier_content=carrier,
    )
    _require(fresh_correct == fresh_wrong,
             f"{case_id} C/W fresh destinations differ")
    evidence = {
        "C": _plan_evidence(correct_plan, destination=False),
        "W": _plan_evidence(wrong_plan, destination=False),
        "F": _plan_evidence(fresh_correct, destination=True),
    }
    _require(evidence["C"]["token_count"] <= MAX_TECHNICAL_SOURCE_TOKENS
             and evidence["W"]["token_count"] <= MAX_TECHNICAL_SOURCE_TOKENS,
             f"{case_id} source exceeds 7000-token bound")
    if case_id == "technical_long":
        low, high = TECHNICAL_LONG_SOURCE_TOKEN_BOUNDS
        _require(low <= evidence["C"]["token_count"] <= high
                 and low <= evidence["W"]["token_count"] <= high,
                 "technical long source is outside frozen 4.5k band")
    selected_ids = {tuple(row["r2_token_ids"]) for row in evidence.values()}
    selected_positions = {
        tuple(row["r2_logical_positions"]) for row in evidence.values()
    }
    _require(len(selected_ids) == len(selected_positions) == 1,
             f"{case_id} C/W/F selected IDs or logical positions differ")
    _require(0 < evidence["F"]["r2_width"] <= 256,
             f"{case_id} R2 width is outside bundle bound")
    core = {
        **{key: value for key, value in material.items()
           if key not in {"correct", "wrong"}},
        "history_sha256": {
            "C": sha256_json(material["correct"]),
            "W": sha256_json(material["wrong"]),
        },
        "plans": evidence,
        "selected_ids_and_positions_identical": True,
        "production_pool_imported": False,
        "production_entropy_requested": False,
        "probe_or_score_present": False,
    }
    result = {
        **core,
        "technical_input_sha256": sha256_json(core),
        "plan_objects": {
            "C": correct_plan,
            "W": wrong_plan,
            "F": fresh_correct,
        },
    }
    return result


__all__ = [
    "E01_RELATIVE_PATH",
    "E01_SHA256",
    "MAX_TECHNICAL_SOURCE_TOKENS",
    "TECHNICAL_CASE_IDS",
    "TECHNICAL_CARRIER",
    "TECHNICAL_INPUT_SCHEMA",
    "TECHNICAL_LONG_SOURCE_TOKEN_BOUNDS",
    "V13TechnicalError",
    "build_technical_case",
    "canonical_json_bytes",
    "load_fixed_e01",
    "sha256_json",
]
