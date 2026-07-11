#!/usr/bin/env python3
"""Build deterministic, model-free review packets for v12 authored cases.

The blind packet deliberately exposes only opaque per-history commitments and
literal messages.  The paired and diversity packets disclose the corresponding
source bindings for independent reviewers who are allowed to see case identity.
This script never imports a tokenizer or model library and never judges a case.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence


DESIGN_ID = "coherent-state-decision-canary-v12"
CASE_SCHEMA = "coherent_state_decision_canary_v12_case_draft_v1"
EXPECTED_CASE_IDS = tuple(f"e{index:02d}" for index in range(1, 7))
VARIANTS = ("correct", "wrong_focal")
PACKET_SCHEMA = "coherent_state_decision_canary_v12_review_packet_v1"
BINDING_SCHEMA = "v12_history_source_binding_v1"
ORDER_SCHEMA = "sha256_seeded_sort_v1"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
COMMON_VISIBLE_CARRIER = {
    "carrier_request": (
        "Write the fixed neutral handoff note for the next assistant. "
        "Output only that note."
    ),
    "carrier_content": (
        "The prior discussion established the operating context and relevant "
        "decision criteria. Continue from this handoff, preserve the existing "
        "constraints, and answer later questions from the state available here. "
        "No unresolved action is introduced by this note."
    ),
    "anchor_user": (
        "Acknowledge receipt of this handoff without adding or repeating any "
        "factual detail."
    ),
    "anchor_assistant": "Acknowledged.",
}


class ReviewPacketError(RuntimeError):
    """A source set or review packet failed a fail-closed invariant."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewPacketError(message)


def _validate_messages(raw: Any, *, case_id: str, variant: str) -> list[dict]:
    _require(isinstance(raw, list) and len(raw) >= 5,
             f"{case_id}/{variant}: messages are incomplete")
    messages = deepcopy(raw)
    for index, message in enumerate(messages):
        _require(isinstance(message, dict) and set(message) == {"role", "content"},
                 f"{case_id}/{variant}: message {index} is not a literal role/content record")
        expected_role = "system" if index == 0 else (
            "user" if index % 2 else "assistant")
        _require(message["role"] == expected_role,
                 f"{case_id}/{variant}: message {index} role is not {expected_role}")
        _require(isinstance(message["content"], str) and message["content"].strip(),
                 f"{case_id}/{variant}: message {index} content is empty")
    return messages


def _validate_metadata_record(raw: dict, key: str, case_id: str) -> dict:
    value = raw.get(key)
    _require(isinstance(value, dict) and value,
             f"{case_id}: {key} metadata is absent")
    return deepcopy(value)


def _require_nonempty_fields(record: dict, fields: Sequence[str], label: str) -> None:
    for field in fields:
        _require(isinstance(record.get(field), str) and record[field].strip() != "",
                 f"{label}: field {field} is empty")


def load_source_cases(paths: Sequence[str | Path]) -> list[dict]:
    """Load exactly e01--e06 and retain byte-level provenance.

    This intentionally performs only tokenizer-free structural checks.  It does
    not substitute for the separately committed production-tokenizer validator.
    """
    _require(len(paths) == len(EXPECTED_CASE_IDS),
             f"expected exactly {len(EXPECTED_CASE_IDS)} paths")
    resolved = [Path(path).resolve() for path in paths]
    _require(len(set(resolved)) == len(resolved), "duplicate source paths")

    records: list[dict] = []
    for path in resolved:
        _require(path.is_file(), f"source is not a file: {path}")
        source_bytes = path.read_bytes()
        try:
            raw = json.loads(source_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReviewPacketError(f"invalid UTF-8 JSON source {path}: {error}") from error
        _require(isinstance(raw, dict), f"source is not a JSON object: {path}")
        case_id = str(raw.get("case_id", ""))
        _require(case_id in EXPECTED_CASE_IDS, f"unexpected case ID {case_id!r}: {path}")
        _require(raw.get("schema") == CASE_SCHEMA, f"{case_id}: case schema differs")
        _require(raw.get("design_id") == DESIGN_ID, f"{case_id}: design ID differs")
        _require(raw.get("status") == "DRAFT_UNREVIEWED",
                 f"{case_id}: status is not DRAFT_UNREVIEWED")
        _require(raw.get("execution_ready") is False,
                 f"{case_id}: source incorrectly authorizes execution")
        _require(raw.get("review") == "PENDING",
                 f"{case_id}: source review is not PENDING")
        _require(raw.get("stratum") == "engineered",
                 f"{case_id}: stratum is not engineered")
        variants = raw.get("variants")
        _require(isinstance(variants, dict) and set(variants) == set(VARIANTS),
                 f"{case_id}: variants must be exactly {VARIANTS}")

        histories = {}
        for variant in VARIANTS:
            variant_record = variants[variant]
            _require(isinstance(variant_record, dict) and
                     set(variant_record) == {"messages"},
                     f"{case_id}/{variant}: variant must contain only messages")
            messages = _validate_messages(
                variant_record["messages"], case_id=case_id, variant=variant)
            histories[variant] = {
                "messages": messages,
                "literal_messages_sha256": _sha256_json(messages),
            }

        _require(histories["correct"]["literal_messages_sha256"] !=
                 histories["wrong_focal"]["literal_messages_sha256"],
                 f"{case_id}: paired histories are byte-identical")
        correct_messages = histories["correct"]["messages"]
        wrong_messages = histories["wrong_focal"]["messages"]
        _require(len(correct_messages) == len(wrong_messages),
                 f"{case_id}: paired message counts differ")
        middle = raw.get("middle_end_msg")
        _require(isinstance(middle, int) and 2 < middle < len(correct_messages),
                 f"{case_id}: middle_end_msg is invalid")
        _require(correct_messages[middle:] == wrong_messages[middle:],
                 f"{case_id}: retained tail differs")
        allow_raw = raw.get("changed_message_allowlist")
        _require(isinstance(allow_raw, list) and allow_raw,
                 f"{case_id}: changed-message allowlist is empty")
        allow = {int(index) for index in allow_raw}
        _require(len(allow) == len(allow_raw) and
                 all(0 < index < middle for index in allow),
                 f"{case_id}: changed-message allowlist is invalid")
        changed = {
            index for index, (correct, wrong) in enumerate(
                zip(correct_messages, wrong_messages))
            if correct != wrong
        }
        _require(changed == allow,
                 f"{case_id}: literal changed messages differ from allowlist")
        focal = _validate_metadata_record(raw, "focal", case_id)
        control = _validate_metadata_record(raw, "nonfocal_control", case_id)
        _require_nonempty_fields(focal, (
            "plant_id", "category", "probe", "correct_target",
            "counterfactual_target", "why_derived", "why_counterfactual_reverses",
        ), f"{case_id}/focal")
        _require_nonempty_fields(control, (
            "plant_id", "category", "probe", "target", "countertarget",
            "why_independent_of_focal",
        ), f"{case_id}/nonfocal_control")
        _require(focal["category"] == "derived_decision",
                 f"{case_id}: focal category differs")
        _require(control["category"] == "unchanged_control",
                 f"{case_id}: nonfocal category differs")
        focal_indices = {
            int(index) for field in (
                "establishing_message_indices", "downstream_reference_indices")
            for index in focal.get(field, [])
        }
        _require(focal_indices and all(0 < index < middle for index in focal_indices),
                 f"{case_id}: focal indices are absent or outside the evicted block")
        control_indices = {
            int(index) for index in control.get("establishing_message_indices", [])
        }
        _require(control_indices and all(
            0 < index < middle for index in control_indices),
            f"{case_id}: nonfocal indices are absent or outside the evicted block")
        _require(not control_indices.intersection(allow),
                 f"{case_id}: nonfocal indices overlap changed messages")
        _require(isinstance(raw.get("title"), str) and raw["title"].strip() != "",
                 f"{case_id}: title is empty")
        _require(isinstance(raw.get("domain"), str) and raw["domain"].strip() != "",
                 f"{case_id}: domain is empty")
        _require(raw.get("length_band") in ("short", "mid"),
                 f"{case_id}: length band differs")
        distractors = raw.get("distractor_fact_inventory")
        _require(isinstance(distractors, list) and len(distractors) >= 2,
                 f"{case_id}: distractor inventory is incomplete")
        retained_purpose = raw.get("retained_tail_purpose")
        _require(isinstance(retained_purpose, str) and retained_purpose.strip() != "",
                 f"{case_id}: retained-tail purpose is empty")
        tokenizer_binding = raw.get("tokenizer_binding")
        _require(isinstance(tokenizer_binding, dict),
                 f"{case_id}: tokenizer binding is absent")
        for field, literal in COMMON_VISIBLE_CARRIER.items():
            if field in tokenizer_binding:
                _require(tokenizer_binding[field] == literal,
                         f"{case_id}: common visible {field} differs")
        records.append({
            "path": str(path),
            "source_file_sha256": _sha256_bytes(source_bytes),
            "case_id": case_id,
            "title": raw.get("title"),
            "domain": raw.get("domain"),
            "length_band": raw.get("length_band"),
            "middle_end_msg": middle,
            "changed_message_allowlist": sorted(allow),
            "authoring_provenance": _validate_metadata_record(
                raw, "authoring_provenance", case_id),
            "focal": focal,
            "nonfocal_control": control,
            "distractor_fact_inventory": deepcopy(distractors),
            "retained_tail_purpose": retained_purpose,
            "histories": histories,
        })

    ids = [record["case_id"] for record in records]
    _require(set(ids) == set(EXPECTED_CASE_IDS) and len(set(ids)) == len(ids),
             f"case coverage differs: {sorted(ids)}")
    source_hashes = [record["source_file_sha256"] for record in records]
    _require(len(set(source_hashes)) == len(source_hashes),
             "duplicate source-file SHA-256 values")
    history_hashes = [
        record["histories"][variant]["literal_messages_sha256"]
        for record in records for variant in VARIANTS
    ]
    _require(len(set(history_hashes)) == len(history_hashes),
             "duplicate literal histories across the source set")
    return sorted(records, key=lambda record: record["case_id"])


def _binding_payload(record: dict, variant: str) -> dict:
    history = record["histories"][variant]
    return {
        "schema": BINDING_SCHEMA,
        "source_file_sha256": record["source_file_sha256"],
        "case_id": record["case_id"],
        "variant": variant,
        "literal_messages_sha256": history["literal_messages_sha256"],
    }


def _bound_history(record: dict, variant: str, *, disclose: bool) -> dict:
    payload = _binding_payload(record, variant)
    result = {
        "binding_commitment_sha256": _sha256_json(payload),
        "messages": deepcopy(record["histories"][variant]["messages"]),
    }
    if disclose:
        result["source_binding"] = payload
    return result


def _packet_base(kind: str, run_id: str, source_set_commitment: str) -> dict:
    return {
        "schema": PACKET_SCHEMA,
        "design_id": DESIGN_ID,
        "packet_kind": kind,
        "packet_run_id": run_id,
        "source_set_commitment_sha256": source_set_commitment,
        "builder_scope": (
            "Deterministic review material only; no tokenizer/model was loaded, "
            "no forward pass occurred, and this packet records no review outcome."
        ),
    }


def _source_set_commitment(records: Sequence[dict]) -> str:
    return _sha256_json([
        {"case_id": record["case_id"],
         "source_file_sha256": record["source_file_sha256"]}
        for record in records
    ])


def build_review_packets(
    paths: Sequence[str | Path], *, seed: str, run_id: str,
) -> dict[str, dict]:
    """Return the three review packets without writing files."""
    _require(isinstance(seed, str) and seed != "", "randomization seed is empty")
    _require(bool(RUN_ID_PATTERN.fullmatch(run_id)), "invalid packet run ID")
    records = load_source_cases(paths)
    source_set_commitment = _source_set_commitment(records)

    sortable = []
    for record in records:
        for variant in VARIANTS:
            history = _bound_history(record, variant, disclose=False)
            order_key = _sha256_bytes(
                b"v12-blind-order\0" + seed.encode("utf-8") + b"\0" +
                history["binding_commitment_sha256"].encode("ascii"))
            sortable.append((order_key, history))
    _require(len({key for key, _history in sortable}) == 12,
             "randomized order-key collision")
    sortable.sort(key=lambda row: row[0])
    blind_entries = []
    for index, (_order_key, history) in enumerate(sortable, start=1):
        blind_entries.append({
            "anonymous_history_id": f"history-{index:02d}",
            **history,
        })
        binding = history["binding_commitment_sha256"]
        matching_record = next(
            record for record in records
            if any(
                _sha256_json(_binding_payload(record, variant)) == binding
                for variant in VARIANTS
            )
        )
        blind_entries[-1]["retained_tail_start_message_index"] = matching_record[
            "middle_end_msg"]
    blind = {
        **_packet_base("blind_singleton", run_id, source_set_commitment),
        "review_boundary": (
            "Review each history independently for naturalness, coherence, dangling "
            "references, conspicuous balancing/padding, fake external actions, and "
            "whether the retained tail remains coherent. Do not consult the paired "
            "or diversity packets while performing this blind review."
        ),
        "anonymization": (
            "Entries disclose no case ID, variant label, target metadata, source path, "
            "or raw source-file hash. Each opaque commitment binds those hidden fields."
        ),
        "common_visible_carrier": deepcopy(COMMON_VISIBLE_CARRIER),
        "carrier_review_instruction": (
            "Review the shared carrier request/content and anchor exchange for target "
            "neutrality, hidden factual implication, and compatibility with every "
            "anonymous history. The carrier is inserted at the disclosed retained-tail "
            "boundary before the retained messages continue."
        ),
        "randomization": {
            "algorithm": ORDER_SCHEMA,
            "seed": seed,
            "seed_sha256": _sha256_bytes(seed.encode("utf-8")),
            "ordered_binding_commitments_sha256": _sha256_json([
                entry["binding_commitment_sha256"] for entry in blind_entries
            ]),
        },
        "histories": blind_entries,
    }

    paired_cases = []
    diversity_cases = []
    for record in records:
        disclosed_histories = {
            variant: _bound_history(record, variant, disclose=True)
            for variant in VARIANTS
        }
        paired_cases.append({
            "case_id": record["case_id"],
            "source_file_sha256": record["source_file_sha256"],
            "title": record["title"],
            "domain": record["domain"],
            "middle_end_msg": record["middle_end_msg"],
            "changed_message_allowlist": record["changed_message_allowlist"],
            "focal": deepcopy(record["focal"]),
            "nonfocal_control": deepcopy(record["nonfocal_control"]),
            "histories": disclosed_histories,
        })
        diversity_cases.append({
            "case_id": record["case_id"],
            "source_file_sha256": record["source_file_sha256"],
            "title": record["title"],
            "domain": record["domain"],
            "length_band": record["length_band"],
            "middle_end_msg": record["middle_end_msg"],
            "changed_message_allowlist": record["changed_message_allowlist"],
            "authoring_provenance": deepcopy(record["authoring_provenance"]),
            "focal": deepcopy(record["focal"]),
            "nonfocal_control": deepcopy(record["nonfocal_control"]),
            "distractor_fact_inventory": deepcopy(record["distractor_fact_inventory"]),
            "retained_tail_purpose": record["retained_tail_purpose"],
            "histories": disclosed_histories,
        })

    paired = {
        **_packet_base("target_aware_paired", run_id, source_set_commitment),
        "review_boundary": (
            "For each pair, verify the focal decision is genuinely derived, the "
            "counterfactual minimally and coherently reverses it, all downstream focal "
            "references are repaired, and the nonfocal chain and retained tail remain "
            "unchanged. Record judgments separately; this packet makes none."
        ),
        "common_visible_carrier": deepcopy(COMMON_VISIBLE_CARRIER),
        "cases": paired_cases,
    }
    diversity = {
        **_packet_base("cross_case_diversity", run_id, source_set_commitment),
        "review_boundary": (
            "Compare all six cases for substantive domain, discourse, decision-rule, "
            "turn-structure, distractor, and probe diversity. Reject shared fill-in-the-"
            "nouns scaffolds. Record judgments separately; this packet makes none."
        ),
        "common_visible_carrier": deepcopy(COMMON_VISIBLE_CARRIER),
        "cases": diversity_cases,
    }

    packets = {
        "blind_singleton": blind,
        "target_aware_paired": paired,
        "cross_case_diversity": diversity,
    }
    _assert_packet_coverage(packets, records)
    return packets


def _iter_packet_histories(packet: dict) -> Iterable[dict]:
    if packet["packet_kind"] == "blind_singleton":
        yield from packet["histories"]
        return
    for case in packet["cases"]:
        for variant in VARIANTS:
            yield case["histories"][variant]


def _assert_packet_coverage(packets: dict[str, dict], records: Sequence[dict]) -> None:
    expected = {
        _sha256_json(_binding_payload(record, variant)):
        record["histories"][variant]["literal_messages_sha256"]
        for record in records for variant in VARIANTS
    }
    _require(len(expected) == 12, "source history commitments are not unique")
    for kind, packet in packets.items():
        observed = [
            history["binding_commitment_sha256"]
            for history in _iter_packet_histories(packet)
        ]
        _require(len(observed) == 12, f"{kind}: omitted or duplicated history count")
        _require(len(set(observed)) == 12, f"{kind}: duplicate history commitments")
        _require(set(observed) == set(expected), f"{kind}: history coverage differs")
        for history in _iter_packet_histories(packet):
            commitment = history["binding_commitment_sha256"]
            _require(_sha256_json(history["messages"]) == expected[commitment],
                     f"{kind}: literal messages do not match their binding")
            if kind != "blind_singleton":
                _require(_sha256_json(history["source_binding"]) == commitment,
                         f"{kind}: disclosed source binding differs")
        if kind != "blind_singleton":
            case_ids = [case["case_id"] for case in packet["cases"]]
            _require(tuple(case_ids) == EXPECTED_CASE_IDS,
                     f"{kind}: case order/coverage differs")


def _seal_packet(packet: dict) -> dict:
    sealed = deepcopy(packet)
    sealed["packet_sha256"] = _sha256_json(packet)
    return sealed


def write_review_packets(packets: dict[str, dict], output_dir: str | Path) -> list[Path]:
    """Write all packets with exclusive-create semantics; never overwrite."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sealed = {kind: _seal_packet(packet) for kind, packet in packets.items()}
    paths = []
    for kind, packet in sealed.items():
        run_id = packet["packet_run_id"]
        path = output_dir / (
            f"coherent_canary_v12_{kind}_{run_id}_{packet['packet_sha256'][:12]}.json"
        )
        _require(not path.exists(), f"refusing to overwrite existing packet: {path}")
        paths.append(path)
    for path, packet in zip(paths, sealed.values()):
        payload = json.dumps(packet, indent=2, ensure_ascii=False) + "\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_paths", nargs=6, help="exactly e01--e06 JSON paths")
    parser.add_argument("--seed", required=True, help="frozen blind-order seed")
    parser.add_argument("--run-id", required=True,
                        help="unique deterministic output run identifier")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets = build_review_packets(
        args.case_paths, seed=args.seed, run_id=args.run_id)
    output_paths = write_review_packets(packets, args.output_dir)
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
