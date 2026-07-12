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
    complete_source_messages,
)


TECHNICAL_INPUT_SCHEMA = "coherent-state-powered-successor-v13-technical-input-v1"
TECHNICAL_PROBE_SCHEMA = "coherent-state-powered-successor-v13-technical-probe-v1"
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

# Exact historical, outcome-seen e01 focal contract transcribed from the sole
# hash-allowlisted E01_RELATIVE_PATH bytes.  No production probe bank or v12
# protocol module is reachable from Stage T.
E01_PROBE = (
    "Under the recorded mandatory selection rule, which ring was selected "
    "for Atlas 4.8? Answer with only the ring name."
)
E01_TARGET_C = "partner beta"
E01_TARGET_W = "staff ring"
E01_PROBE_SHA256 = "16f995d86894f155a0682ee1a148d33f0d2c17ceac2571b26313f38fe3cbcb9c"
E01_TARGET_C_SHA256 = "087a876aa49a653060fc19d92823f59cd3870cbfe253d9dd88c8f67d271d0b56"
E01_TARGET_W_SHA256 = "b10d04f68db4f0d047c634a577dd2f37375ae75b1dca201c664315bb95faed59"

# The long-geometry fact is in TECHNICAL_CARRIER, which remains visible and
# byte-identical in C, W, and the fresh compact destination.  The long filler
# is deliberately not used: it lies in the evicted middle and is absent in F.
TECHNICAL_LONG_PROBE = (
    "What does the handoff preserve for the next assistant? Answer with only "
    "one word."
)
TECHNICAL_LONG_TARGET = "continuity"
TECHNICAL_LONG_COUNTERTARGET = "criteria"
TECHNICAL_LONG_PROBE_SHA256 = "2d3b063bdb5b8acf57da54a6b6893e82c29aa33a7dd74287732d798962c361ee"
TECHNICAL_LONG_TARGET_SHA256 = "0adc36057cef79f8a7b8a862bb9fab8cbdc13828498c636c4d92d761fd621a0f"
TECHNICAL_LONG_COUNTERTARGET_SHA256 = "262d517bffd06484e2341b4a379c7f4fa4be9284a9a82631c03420953d60bd67"


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


def _literal_sha256(value: str, expected: str, label: str) -> str:
    observed = sha256_bytes(value.encode("utf-8"))
    _require(observed == expected, f"{label} literal hash differs")
    return observed


def _technical_probe_contract(case_id: str) -> dict[str, Any]:
    _require(case_id in TECHNICAL_CASE_IDS, "unknown technical case ID")
    if case_id == "technical_e01":
        probe = E01_PROBE
        correct = E01_TARGET_C
        counter = E01_TARGET_W
        probe_sha = _literal_sha256(
            probe, E01_PROBE_SHA256, "technical e01 probe")
        correct_sha = _literal_sha256(
            correct, E01_TARGET_C_SHA256, "technical e01 correct target")
        counter_sha = _literal_sha256(
            counter, E01_TARGET_W_SHA256,
            "technical e01 counterfactual target")
        provenance = "outcome_seen_hash_allowlisted_e01_focal"
        visible_origin = "historical_e01"
    else:
        probe = TECHNICAL_LONG_PROBE
        correct = TECHNICAL_LONG_TARGET
        counter = TECHNICAL_LONG_COUNTERTARGET
        probe_sha = _literal_sha256(
            probe, TECHNICAL_LONG_PROBE_SHA256, "technical long probe")
        correct_sha = _literal_sha256(
            correct, TECHNICAL_LONG_TARGET_SHA256,
            "technical long correct target")
        counter_sha = _literal_sha256(
            counter, TECHNICAL_LONG_COUNTERTARGET_SHA256,
            "technical long countertarget")
        _require(correct in TECHNICAL_CARRIER and counter in TECHNICAL_CARRIER,
                 "technical long targets are not both carrier-visible")
        provenance = "neutral_byte_visible_carrier_fact"
        visible_origin = "technical_carrier_all_histories"
    core = {
        "schema": TECHNICAL_PROBE_SCHEMA,
        "design_id": DESIGN_ID,
        "case_id": case_id,
        "probe": probe,
        "probe_sha256": probe_sha,
        "correct_target": correct,
        "correct_target_sha256": correct_sha,
        "counterfactual_target": counter,
        "counterfactual_target_sha256": counter_sha,
        "provenance": provenance,
        "visible_origin": visible_origin,
        "semantic_n": 0,
        "inferential_use": "FORBIDDEN",
    }
    return {**core, "technical_probe_sha256": sha256_json(core)}


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
    focal = value.get("focal")
    _require(isinstance(focal, Mapping)
             and focal.get("probe") == E01_PROBE
             and focal.get("correct_target") == E01_TARGET_C
             and focal.get("counterfactual_target") == E01_TARGET_W,
             "technical e01 focal probe contract differs")
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
    probe_contract = _technical_probe_contract(case_id)
    stable_id = sha256_json([
        DESIGN_ID,
        case_id,
        E01_SHA256,
        sha256_bytes(TECHNICAL_CARRIER.encode("utf-8")),
        TECHNICAL_LONG_FILLER_UNIT if case_id == "technical_long" else None,
        TECHNICAL_LONG_FILLER_REPEATS if case_id == "technical_long" else None,
        probe_contract["technical_probe_sha256"],
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
        "technical_probe": probe_contract,
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
    completed_correct = complete_source_messages(
        material["correct"], middle_end_msg=middle,
        carrier_content=carrier)
    completed_wrong = complete_source_messages(
        material["wrong"], middle_end_msg=middle,
        carrier_content=carrier)
    compact_correct = deepcopy(completed_correct[:1]) + deepcopy(
        completed_correct[middle:])
    compact_wrong = deepcopy(completed_wrong[:1]) + deepcopy(
        completed_wrong[middle:])
    _require(compact_correct == compact_wrong,
             f"{case_id} C/W compact visible messages differ")
    context_messages = {
        "C": completed_correct,
        "W": completed_wrong,
        "F": compact_correct,
    }
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
           if key not in {"correct", "wrong", "technical_probe"}},
        "history_sha256": {
            "C": sha256_json(material["correct"]),
            "W": sha256_json(material["wrong"]),
        },
        "context_messages_sha256": {
            history: sha256_json(messages)
            for history, messages in context_messages.items()
        },
        "plans": evidence,
        "selected_ids_and_positions_identical": True,
        "production_pool_imported": False,
        "production_entropy_requested": False,
        "production_probe_reachable": False,
        "score_values_present": False,
        "technical_probe_sha256": material["technical_probe"][
            "technical_probe_sha256"],
    }
    technical_input_sha256 = sha256_json(core)
    result = {
        **core,
        "technical_input_sha256": technical_input_sha256,
        "technical_probe": material["technical_probe"],
        "technical_case_sha256": sha256_json({
            "technical_input_sha256": technical_input_sha256,
            "technical_probe_sha256": material["technical_probe"][
                "technical_probe_sha256"],
        }),
        "plan_objects": {
            "C": correct_plan,
            "W": wrong_plan,
            "F": fresh_correct,
        },
        "context_messages": context_messages,
    }
    return result


__all__ = [
    "E01_RELATIVE_PATH",
    "E01_SHA256",
    "MAX_TECHNICAL_SOURCE_TOKENS",
    "TECHNICAL_CASE_IDS",
    "TECHNICAL_CARRIER",
    "TECHNICAL_INPUT_SCHEMA",
    "TECHNICAL_PROBE_SCHEMA",
    "TECHNICAL_LONG_SOURCE_TOKEN_BOUNDS",
    "V13TechnicalError",
    "build_technical_case",
    "canonical_json_bytes",
    "load_fixed_e01",
    "sha256_json",
]
