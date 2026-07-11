"""Pure case construction for the coherent-summary-state experiment.

No model, tokenizer, or tensor code belongs here.  Keeping history substitution,
probe selection, and target validation pure makes the causal layout auditable
before any cache manipulation is attempted.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from arms_common import SUMMARY_REQUEST


FROZEN_ORDER = (
    "c10", "c02", "c01", "c04", "c07", "c11",
    "c05", "c09", "c06", "c12", "c08", "c03",
)

WRONG_DONOR = {
    "c10": "c13", "c02": "c14", "c01": "c15",
    "c04": "c16", "c07": "c17", "c11": "c18",
    "c05": "c25", "c09": "c26", "c06": "c27",
    "c12": "c28", "c08": "c29", "c03": "c30",
}

PRIMARY_CATEGORIES = ("referent", "sense")


class CaseConstructionError(RuntimeError):
    """A frozen causal-case invariant was violated."""


def load_scenarios(path: str | Path) -> dict[str, dict]:
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list):
        raise CaseConstructionError("scenario file must contain a JSON list")
    by_id = {str(row.get("id")): row for row in rows}
    if len(by_id) != len(rows) or "None" in by_id:
        raise CaseConstructionError("scenario IDs must be present and unique")
    missing = set(FROZEN_ORDER) - set(by_id)
    if missing:
        raise CaseConstructionError(f"frozen scenarios missing: {sorted(missing)}")
    return by_id


def frozen_scenarios(by_id: dict[str, dict], n: int = 12) -> list[dict]:
    if not 1 <= n <= len(FROZEN_ORDER):
        raise CaseConstructionError(f"invalid frozen scenario count: {n}")
    return [deepcopy(by_id[cid]) for cid in FROZEN_ORDER[:n]]


def select_primary_plants(scenario: dict) -> list[dict]:
    """Return the first referent and first sense plant, in that order."""
    selected = []
    for category in PRIMARY_CATEGORIES:
        match = next((p for p in scenario.get("plants", [])
                      if p.get("category") == category), None)
        if match is None:
            raise CaseConstructionError(
                f"{scenario.get('id')} has no {category} plant")
        for field in ("id", "probe", "gold"):
            if not str(match.get(field, "")).strip():
                raise CaseConstructionError(
                    f"{scenario.get('id')} {category} plant lacks {field}")
        selected.append(deepcopy(match))
    return selected


def validate_native_conversation(conv: dict) -> None:
    cid = str(conv.get("id", ""))
    msgs = conv.get("messages")
    if not cid or not isinstance(msgs, list) or len(msgs) < 3:
        raise CaseConstructionError("native conversation is incomplete")
    if msgs[0].get("role") != "system":
        raise CaseConstructionError(f"{cid}: first message is not system")
    tsm = conv.get("sections", {}).get("middle_end_msg")
    if not isinstance(tsm, int) or not 1 < tsm < len(msgs):
        raise CaseConstructionError(f"{cid}: invalid middle_end_msg={tsm}")
    if msgs[tsm].get("role") != "user":
        raise CaseConstructionError(f"{cid}: retained tail must begin with user")
    for i in range(1, len(msgs)):
        expected = "user" if i % 2 else "assistant"
        if msgs[i].get("role") != expected:
            raise CaseConstructionError(
                f"{cid}: message {i} role {msgs[i].get('role')} != {expected}")


def correct_source_messages(conv: dict, request: str = SUMMARY_REQUEST) -> list[dict]:
    validate_native_conversation(conv)
    return deepcopy(conv["messages"]) + [{"role": "user", "content": request}]


def fresh_source_messages(conv: dict, request: str = SUMMARY_REQUEST) -> list[dict]:
    validate_native_conversation(conv)
    return [deepcopy(conv["messages"][0]), {"role": "user", "content": request}]


def wrong_source_messages(target: dict, donor: dict,
                          request: str = SUMMARY_REQUEST) -> list[dict]:
    """Target system/tail with donor's evicted message block, then request.

    The donor block begins after its system message and ends immediately before
    its retained tail.  This preserves role alternation while preventing donor
    system or tail text from entering the source.
    """
    validate_native_conversation(target)
    validate_native_conversation(donor)
    expected = WRONG_DONOR.get(str(target["id"]))
    if expected is not None and donor["id"] != expected:
        raise CaseConstructionError(
            f"{target['id']}: donor {donor['id']} != frozen donor {expected}")
    tt = target["sections"]["middle_end_msg"]
    dt = donor["sections"]["middle_end_msg"]
    msgs = [deepcopy(target["messages"][0])]
    msgs.extend(deepcopy(donor["messages"][1:dt]))
    msgs.extend(deepcopy(target["messages"][tt:]))
    msgs.append({"role": "user", "content": request})
    # This synthetic source has no sections, so validate its roles directly.
    for i, msg in enumerate(msgs[1:], 1):
        expected_role = "user" if i % 2 else "assistant"
        if msg.get("role") != expected_role:
            raise CaseConstructionError(
                f"{target['id']}: wrong-source role break at {i}")
    return msgs


def compacted_messages(conv: dict, summary_text: str,
                       request: str = SUMMARY_REQUEST) -> list[dict]:
    """Exact preregistered destination, without the legacy context-note preamble."""
    validate_native_conversation(conv)
    if not summary_text:
        raise CaseConstructionError("summary text is empty")
    tsm = conv["sections"]["middle_end_msg"]
    return [
        deepcopy(conv["messages"][0]),
        {"role": "user", "content": request},
        {"role": "assistant", "content": summary_text},
        *deepcopy(conv["messages"][tsm:]),
    ]


def _target_rows(raw: object) -> Iterable[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("targets"), list):
        return raw["targets"]
    raise CaseConstructionError("target file must be a list or {targets: [...]}")


def load_and_validate_targets(path: str | Path,
                              scenarios: dict[str, dict]) -> dict[str, dict]:
    """Validate one precommitted counterfactual for every frozen primary plant."""
    raw = json.loads(Path(path).read_text())
    rows = list(_target_rows(raw))
    by_plant: dict[str, dict] = {}
    for row in rows:
        pid = str(row.get("plant_id", ""))
        if not pid or pid in by_plant:
            raise CaseConstructionError(f"target plant IDs must be unique: {pid!r}")
        for field in ("correct", "counterfactual", "basis"):
            if not str(row.get(field, "")).strip():
                raise CaseConstructionError(f"{pid}: missing target field {field}")
        if row["correct"].strip() == row["counterfactual"].strip():
            raise CaseConstructionError(f"{pid}: targets are identical")
        by_plant[pid] = deepcopy(row)

    expected: dict[str, dict] = {}
    for cid in FROZEN_ORDER:
        for plant in select_primary_plants(scenarios[cid]):
            expected[plant["id"]] = plant
    missing = set(expected) - set(by_plant)
    extra = set(by_plant) - set(expected)
    if missing or extra:
        raise CaseConstructionError(
            f"target coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    # The scientific collaborator froze concise answer phrases derived from the
    # scaffold gold, rather than copying its sometimes explanatory prose byte for
    # byte. Preserve the exact source and hash in memory without mutating the
    # frozen target file.
    for pid, plant in expected.items():
        by_plant[pid]["scaffold_gold"] = plant["gold"]
        by_plant[pid]["scaffold_gold_sha256"] = hashlib.sha256(
            plant["gold"].encode()).hexdigest()
    return by_plant
