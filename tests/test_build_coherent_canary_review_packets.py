from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_coherent_canary_review_packets.py"
SPEC = importlib.util.spec_from_file_location("v12_review_packets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _messages(number: int, variant: str) -> list[dict]:
    changed = "alpha" if variant == "correct" else "omega"
    return [
        {"role": "system", "content": f"Coordinator protocol {number}."},
        {"role": "user", "content": f"A threshold observation was {changed}."},
        {"role": "assistant", "content": f"I recorded {changed} for the decision."},
        {"role": "user", "content": f"Unchanged control question {number}."},
        {"role": "assistant", "content": f"Unchanged control answer {number}."},
        {"role": "user", "content": f"Retained neutral question {number}."},
        {"role": "assistant", "content": f"Retained neutral answer {number}."},
    ]


def _case(number: int) -> dict:
    case_id = f"e{number:02d}"
    return {
        "schema": MODULE.CASE_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": case_id,
        "status": "DRAFT_UNREVIEWED",
        "execution_ready": False,
        "review": "PENDING",
        "stratum": "engineered",
        "title": f"Fixture title {number}",
        "domain": f"fixture domain {number}",
        "length_band": "short" if number != 4 else "mid",
        "middle_end_msg": 5,
        "changed_message_allowlist": [1, 2],
        "authoring_provenance": {"session": f"author-{number}"},
        "focal": {
            "plant_id": f"focal-{number}",
            "category": "derived_decision",
            "probe": f"Focal question {number}?",
            "correct_target": f"choice-{number}-a",
            "counterfactual_target": f"choice-{number}-b",
            "establishing_message_indices": [1],
            "downstream_reference_indices": [2],
            "why_derived": "It combines the rule and observation.",
            "why_counterfactual_reverses": "The changed observation reverses it.",
        },
        "nonfocal_control": {
            "plant_id": f"control-{number}",
            "category": "unchanged_control",
            "probe": f"Control question {number}?",
            "target": f"control-{number}",
            "countertarget": f"other-{number}",
            "establishing_message_indices": [3, 4],
            "why_independent_of_focal": "It is a separate recorded fact.",
        },
        "distractor_fact_inventory": [f"fact-{number}-a", f"fact-{number}-b"],
        "retained_tail_purpose": f"neutral continuation {number}",
        "tokenizer_binding": deepcopy(MODULE.COMMON_VISIBLE_CARRIER),
        "variants": {
            variant: {"messages": _messages(number, variant)}
            for variant in MODULE.VARIANTS
        },
    }


def _write_cases(tmp_path: Path) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for number in range(1, 7):
        path = tmp_path / f"source-{number}.json"
        path.write_text(json.dumps(_case(number)), encoding="utf-8")
        paths.append(path)
    return paths


def _commitments(packet: dict) -> list[str]:
    return [row["binding_commitment_sha256"] for row in packet["histories"]]


def test_build_is_deterministic_complete_and_blind(tmp_path: Path) -> None:
    paths = _write_cases(tmp_path)
    first = MODULE.build_review_packets(paths, seed="frozen-seed", run_id="review-01")
    second = MODULE.build_review_packets(
        list(reversed(paths)), seed="frozen-seed", run_id="review-01")
    assert first == second

    blind = first["blind_singleton"]
    assert [row["anonymous_history_id"] for row in blind["histories"]] == [
        f"history-{index:02d}" for index in range(1, 13)
    ]
    assert len(set(_commitments(blind))) == 12
    assert all(set(row) == {
        "anonymous_history_id", "binding_commitment_sha256", "messages",
        "retained_tail_start_message_index",
    } for row in blind["histories"])
    forbidden = {"case_id", "variant", "source_file_sha256", "source_binding",
                 "focal", "nonfocal_control", "target", "countertarget"}
    assert all(not forbidden.intersection(row) for row in blind["histories"])

    expected_messages = {
        json.dumps(_messages(number, variant), sort_keys=True)
        for number in range(1, 7) for variant in MODULE.VARIANTS
    }
    observed_messages = {
        json.dumps(row["messages"], sort_keys=True) for row in blind["histories"]
    }
    assert observed_messages == expected_messages
    assert blind["common_visible_carrier"] == MODULE.COMMON_VISIBLE_CARRIER
    assert all(row["retained_tail_start_message_index"] == 5
               for row in blind["histories"])


def test_disclosed_packets_open_every_blind_binding(tmp_path: Path) -> None:
    packets = MODULE.build_review_packets(
        _write_cases(tmp_path), seed="seed-a", run_id="review-02")
    blind_commitments = set(_commitments(packets["blind_singleton"]))
    for kind in ("target_aware_paired", "cross_case_diversity"):
        packet = packets[kind]
        assert [case["case_id"] for case in packet["cases"]] == list(
            MODULE.EXPECTED_CASE_IDS)
        opened = set()
        for case in packet["cases"]:
            for variant in MODULE.VARIANTS:
                history = case["histories"][variant]
                payload = history["source_binding"]
                assert payload["source_file_sha256"] == case["source_file_sha256"]
                assert payload["case_id"] == case["case_id"]
                assert payload["variant"] == variant
                assert MODULE._sha256_json(payload) == history[
                    "binding_commitment_sha256"]
                opened.add(history["binding_commitment_sha256"])
        assert opened == blind_commitments


def test_seed_commits_order_and_changes_it(tmp_path: Path) -> None:
    paths = _write_cases(tmp_path)
    left = MODULE.build_review_packets(paths, seed="seed-left", run_id="r")
    right = MODULE.build_review_packets(paths, seed="seed-right", run_id="r")
    assert _commitments(left["blind_singleton"]) != _commitments(
        right["blind_singleton"])
    randomization = left["blind_singleton"]["randomization"]
    assert randomization["seed"] == "seed-left"
    assert randomization["seed_sha256"] == MODULE._sha256_bytes(b"seed-left")
    assert randomization["ordered_binding_commitments_sha256"] == MODULE._sha256_json(
        _commitments(left["blind_singleton"])
    )


@pytest.mark.parametrize("mutation,match", [
    ("omit", "exactly 6 paths"),
    ("duplicate_case", "case coverage differs"),
    ("duplicate_history", "paired histories are byte-identical"),
])
def test_source_coverage_failures_are_closed(
    tmp_path: Path, mutation: str, match: str,
) -> None:
    paths = _write_cases(tmp_path)
    if mutation == "omit":
        paths.pop()
    elif mutation == "duplicate_case":
        raw = json.loads(paths[-1].read_text())
        raw["case_id"] = "e05"
        paths[-1].write_text(json.dumps(raw))
    else:
        raw = json.loads(paths[0].read_text())
        raw["variants"]["wrong_focal"] = deepcopy(raw["variants"]["correct"])
        paths[0].write_text(json.dumps(raw))
    with pytest.raises(MODULE.ReviewPacketError, match=match):
        MODULE.build_review_packets(paths, seed="seed", run_id="run")


def test_writes_content_sealed_unique_files_and_refuses_overwrite(tmp_path: Path) -> None:
    paths = _write_cases(tmp_path / "sources")
    packets = MODULE.build_review_packets(paths, seed="seed", run_id="run-03")
    output_dir = tmp_path / "packets"
    written = MODULE.write_review_packets(packets, output_dir)
    assert len(written) == 3
    assert len({path.name for path in written}) == 3
    for path in written:
        packet = json.loads(path.read_text())
        packet_sha = packet.pop("packet_sha256")
        assert packet_sha == MODULE._sha256_json(packet)
        assert packet_sha[:12] in path.name
    with pytest.raises(MODULE.ReviewPacketError, match="refusing to overwrite"):
        MODULE.write_review_packets(packets, output_dir)


def test_builder_has_no_tokenizer_or_model_dependency() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "transformers" not in source
    assert "torch" not in source
    assert "mlx" not in source
    assert "coherent_canary_stimuli" not in source
