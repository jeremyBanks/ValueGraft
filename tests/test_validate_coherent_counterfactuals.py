from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_coherent_counterfactuals",
    ROOT / "scripts" / "validate_coherent_counterfactuals.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

INPUT_DIR = ROOT / "data" / "coherent_state_counterfactuals" / "banked_c10_c02"
CANDIDATE_PATH = INPUT_DIR / "c02_referent_unreviewed.json"


@pytest.fixture(scope="session")
def tokenizer():
    return MODULE._load_tokenizer()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_fixture(tokenizer):
    candidate = json.loads(CANDIDATE_PATH.read_text())
    base_path = ROOT / candidate["base_binding"]["path"]
    scenario_path = ROOT / candidate["plant_target_binding"]["scenario_path"]
    target_path = ROOT / candidate["plant_target_binding"]["target_path"]
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    return candidate, {
        "candidate_path": CANDIDATE_PATH.relative_to(ROOT).as_posix(),
        "candidate_raw_sha256": _sha(CANDIDATE_PATH),
        "base": json.loads(base_path.read_text()),
        "base_path": base_path.relative_to(ROOT).as_posix(),
        "base_raw_sha256": _sha(base_path),
        "scenarios": json.loads(scenario_path.read_text()),
        "scenario_path": scenario_path.relative_to(ROOT).as_posix(),
        "scenario_raw_sha256": _sha(scenario_path),
        "targets": json.loads(target_path.read_text()),
        "targets_path": target_path.relative_to(ROOT).as_posix(),
        "targets_raw_sha256": _sha(target_path),
        "tokenizer": tokenizer,
        "expected_provenance": manifest["authoring_provenance"],
    }


def _set_path(value, path, replacement):
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def test_committed_banked_candidates_mechanically_pass_but_never_authorize(tokenizer):
    result = MODULE.validate_directory(ROOT, tokenizer=tokenizer)
    MODULE._require_sealed(result)
    assert result["verdict"] == "MECHANICAL_PASS"
    assert result["candidate_count"] == 4
    assert result["mechanical_pass"] is True
    assert result["semantic_authorized"] is False
    assert result["execution_authorized"] is False
    assert result["model_forward_performed"] is False
    assert result["embedded_candidate_review_status"] == "PENDING"
    assert result["external_review_status"] == "FAIL"
    assert result["missing_review_attestations"] == []
    assert result["external_review_evidence"]["status"] == "FAIL"
    assert result["external_review_evidence"]["blind_review"][
        "overall_verdict"] == "FAIL"
    assert result["external_review_evidence"]["target_aware_factual_review"][
        "overall_verdict"] == "REVISE_ALL_FOUR"
    assert all(row["execution_authorized"] is False
               for row in result["external_review_evidence"][
                   "candidate_verdicts"])
    inventory_paths = {row["path"] for row in result["committed_input_inventory"]}
    assert MODULE.VALIDATOR_PATH in inventory_paths
    assert MODULE.SOURCE_NOTE_PATH in inventory_paths
    assert (f"{MODULE.DEFAULT_INPUT_DIR}/reviews/"
            f"{MODULE.BLIND_REVIEW_NAME}") in inventory_paths
    assert (f"{MODULE.DEFAULT_INPUT_DIR}/reviews/"
            f"{MODULE.FACTUAL_REVIEW_NAME}") in inventory_paths
    assert {(row["conversation_id"], row["category"])
            for row in result["candidates"]} == MODULE.EXPECTED_COVERAGE
    assert all(row["mechanical_status"] == "MECHANICAL_PASS" and
               row["semantic_authorized"] is False and
               row["execution_authorized"] is False and
               row["exact_decoded_roundtrip"] is True
               for row in result["candidates"])


CANDIDATE_MUTATIONS = [
    (("schema",), "wrong-schema"),
    (("status",), "REVIEWED"),
    (("execution_ready",), True),
    (("candidate_id",), "banked-wrong"),
    (("authoring_provenance", "author"), "Other"),
    (("authoring_provenance", "model_id"), "wrong-model"),
    (("authoring_provenance", "source_note_path"), "notes/wrong.md"),
    (("authoring_provenance", "source_note_sha256"), "0" * 64),
    (("authoring_provenance", "warning"), "safe"),
    (("base_binding", "path"), "data/synthetic/c10.json"),
    (("base_binding", "raw_file_sha256"), "0" * 64),
    (("base_binding", "conversation_id"), "c10"),
    (("base_binding", "complete_base_conversation_canonical_sha256"), "0" * 64),
    (("base_binding", "middle_end_msg"), 34),
    (("plant_target_binding", "plant_id"), "c02-sense-1"),
    (("plant_target_binding", "category"), "stance"),
    (("plant_target_binding", "scenario_path"), "data/wrong.json"),
    (("plant_target_binding", "scenario_raw_file_sha256"), "0" * 64),
    (("plant_target_binding", "scenario_plant", "probe"), "wrong probe"),
    (("plant_target_binding", "scenario_plant_canonical_sha256"), "0" * 64),
    (("plant_target_binding", "banked_conversation_plant", "probe"), "wrong probe"),
    (("plant_target_binding", "banked_conversation_plant_canonical_sha256"), "0" * 64),
    (("plant_target_binding", "target_path"), "data/wrong.json"),
    (("plant_target_binding", "target_raw_file_sha256"), "0" * 64),
    (("plant_target_binding", "target_row", "counterfactual"), "wrong target"),
    (("plant_target_binding", "target_row_canonical_sha256"), "0" * 64),
    (("changed_message_allowlist", 0, "message_index"), 30),
    (("changed_message_allowlist", 0, "role"), "assistant"),
    (("changed_message_allowlist", 0, "base_content_sha256"), "0" * 64),
    (("changed_message_allowlist", 0, "counterfactual_content_sha256"), "0" * 64),
    (("changed_message_allowlist", 0, "base_content_token_count"), 999),
    (("changed_message_allowlist", 0, "counterfactual_content_token_count"), 999),
    (("unchanged_message_indices",), []),
    (("counterfactual_conversation", "title"), "wrong title"),
    (("counterfactual_conversation", "messages", 0, "content"), "wrong system"),
    (("counterfactual_conversation", "messages", 5, "content"), "unapproved"),
    (("counterfactual_conversation", "messages", 33, "content"), "wrong tail"),
    (("counterfactual_conversation", "messages", 31, "role"), "assistant"),
    (("counterfactual_conversation", "messages", 31, "content"), "changed again"),
    (("counterfactual_conversation_canonical_sha256",), "0" * 64),
    (("tokenizer_binding", "model_id"), "wrong/model"),
    (("tokenizer_binding", "revision"), "0" * 40),
    (("tokenizer_binding", "tokenizer_class_observed"), "WrongTokenizer"),
    (("tokenizer_binding", "summary_request_sha256"), "0" * 64),
    (("tokenizer_binding", "chat_template_sha256"), "0" * 64),
    (("observed_banked_mechanical_validation", "status"), "PASS"),
    (("observed_banked_mechanical_validation", "correct_prefix_token_count"), 1),
    (("observed_banked_mechanical_validation", "counterfactual_prefix_token_count"), 1),
    (("observed_banked_mechanical_validation", "prefix_token_counts_equal"), False),
    (("observed_banked_mechanical_validation", "correct_prefix_token_ids_sha256"), "0" * 64),
    (("observed_banked_mechanical_validation", "counterfactual_prefix_token_ids_sha256"), "0" * 64),
    (("observed_banked_mechanical_validation", "differing_prefix_token_position_count"), 0),
    (("observed_banked_mechanical_validation", "correct_message_start_positions"), []),
    (("observed_banked_mechanical_validation", "counterfactual_message_start_positions"), []),
    (("observed_banked_mechanical_validation", "message_start_positions_equal"), False),
    (("observed_banked_mechanical_validation", "turn_aligned_replay_call_widths"), []),
    (("observed_banked_mechanical_validation", "ordinary_4096_call_widths"), []),
    (("observed_banked_mechanical_validation", "summary_logical_start"), 0),
    (("observed_banked_mechanical_validation", "changed_messages_strictly_evicted"), False),
    (("observed_banked_mechanical_validation", "all_changed_message_content_token_counts_equal"), False),
    (("review", "overall_status"), "PASS"),
    (("review", "factual_counterfactual_audit", "status"), "PASS"),
    (("review", "blind_naturalness_coherence_review", "reviewer"), "someone"),
    (("review", "adjudication", "status"), "PASS"),
]


@pytest.mark.parametrize(("path", "replacement"), CANDIDATE_MUTATIONS)
def test_each_candidate_contract_field_is_load_bearing(
        tokenizer, path, replacement):
    candidate, kwargs = _candidate_fixture(tokenizer)
    _set_path(candidate, path, replacement)
    with pytest.raises(MODULE.CounterfactualValidationError):
        MODULE._validate_candidate_doc(candidate, **kwargs)


@pytest.mark.parametrize("section", [
    "base_binding", "plant_target_binding", "changed_message_allowlist",
    "tokenizer_binding", "observed_banked_mechanical_validation", "review",
])
def test_nested_schema_shape_is_exact(tokenizer, section):
    candidate, kwargs = _candidate_fixture(tokenizer)
    target = candidate[section]
    if isinstance(target, list):
        target[0]["unexpected"] = True
    else:
        target["unexpected"] = True
    with pytest.raises(MODULE.CounterfactualValidationError, match="fields differ"):
        MODULE._validate_candidate_doc(candidate, **kwargs)


def test_changed_message_must_be_strictly_before_middle_end(tokenizer):
    candidate, kwargs = _candidate_fixture(tokenizer)
    candidate["changed_message_allowlist"][0]["message_index"] = 33
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="strictly evicted"):
        MODULE._validate_candidate_doc(candidate, **kwargs)


def test_special_token_and_template_control_strings_are_rejected(tokenizer):
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="special token ID"):
        MODULE._validate_content_safety(tokenizer, "<|im_start|>", "mutant")
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="template control string"):
        MODULE._validate_content_safety(tokenizer, "ordinary {% text", "mutant")


def test_exact_decoded_round_trip_is_required():
    class NonRoundTripTokenizer:
        def encode(self, _text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7]

        def decode(self, _ids, **_kwargs):
            return "different"

    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="decoded round-trip differs"):
        MODULE._require_exact_roundtrip(
            NonRoundTripTokenizer(), "original", "mutant")


def test_validator_rendering_matches_production_prefix_path_for_every_candidate(
        tokenizer):
    sys.path.insert(0, str(ROOT / "src"))
    from coherent_state_tokens import generation_prefix_ids

    for path in sorted(INPUT_DIR.glob("*_unreviewed.json")):
        candidate = json.loads(path.read_text())
        messages = list(candidate["counterfactual_conversation"]["messages"]) + [{
            "role": "user", "content": MODULE.SUMMARY_REQUEST,
        }]
        assert MODULE._generation_prefix_ids(tokenizer, messages) == \
            generation_prefix_ids(tokenizer, messages)


MANIFEST_HEADER_MUTATIONS = [
    (("schema",), "wrong"),
    (("status",), "REVIEWED"),
    (("execution_ready",), True),
    (("scope",), "wrong"),
    (("authoring_provenance", "author"), "Other"),
    (("authoring_provenance", "model_id"), "wrong"),
    (("authoring_provenance", "source_note_path"), "notes/wrong.md"),
    (("authoring_provenance", "source_note_sha256"), "0" * 64),
    (("warning",), "wrong"),
    (("expected_candidate_count",), 3),
    (("candidate_count",), 3),
    (("all_candidates_unreviewed",), False),
    (("all_candidates_execution_ready_false",), False),
    (("review_gate", "status"), "PASS"),
    (("review_gate", "requirements"), []),
]


@pytest.mark.parametrize(("path", "replacement"), MANIFEST_HEADER_MUTATIONS)
def test_each_manifest_header_field_is_load_bearing(path, replacement):
    manifest_path = INPUT_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    note_sha = _sha(ROOT / manifest["authoring_provenance"]["source_note_path"])
    _set_path(manifest, path, replacement)
    with pytest.raises(MODULE.CounterfactualValidationError):
        MODULE._validate_manifest_header(manifest, note_sha)


@pytest.mark.parametrize("field", sorted(MODULE.MANIFEST_ROW_KEYS))
def test_each_manifest_candidate_row_field_is_load_bearing(field):
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    expected = manifest["candidates"][0]
    mutant = copy.deepcopy(expected)
    mutant[field] = "mutated"
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="manifest row differs"):
        MODULE._validate_manifest_row(mutant, expected)


@pytest.mark.parametrize("duplicate_field", ["path", "candidate_id", "plant_id"])
def test_manifest_coverage_must_be_unique(duplicate_field):
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    rows = copy.deepcopy(manifest["candidates"])
    rows[1][duplicate_field] = rows[0][duplicate_field]
    actual = [row["path"] for row in manifest["candidates"]]
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="coverage is not unique"):
        MODULE._validate_manifest_row_inventory(
            rows, MODULE.DEFAULT_INPUT_DIR, actual)


def test_manifest_must_cover_every_candidate_file_once():
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    actual = [row["path"] for row in manifest["candidates"]][:-1]
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="manifest/file candidate coverage differs"):
        MODULE._validate_manifest_row_inventory(
            manifest["candidates"], MODULE.DEFAULT_INPUT_DIR, actual)


def test_working_bytes_must_equal_committed_bytes(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=tmp_path, check=True)
    path = tmp_path / "bound.json"
    path.write_text('{"value":1}\n')
    subprocess.run(["git", "add", "bound.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=tmp_path, check=True)
    path.write_text('{"value":2}\n')
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="working bytes differ from committed bytes"):
        MODULE._committed_bytes(tmp_path, "bound.json")


def test_sealed_output_is_exact_round_trip_and_never_overwritten(tmp_path):
    output = tmp_path / "result.json"
    document = MODULE._seal({
        "schema": MODULE.OUTPUT_SCHEMA,
        "verdict": "MECHANICAL_PASS",
        "mechanical_pass": True,
        "semantic_authorized": False,
        "execution_authorized": False,
        "model_forward_performed": False,
    })
    MODULE._write_sealed_unique(output, document)
    assert json.loads(output.read_text()) == document
    MODULE._require_sealed(json.loads(output.read_text()))
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="refusing to overwrite"):
        MODULE._write_sealed_unique(output, document)

    tampered = MODULE._seal({
        **document,
        "semantic_authorized": True,
        "execution_authorized": True,
    })
    with pytest.raises(MODULE.CounterfactualValidationError,
                       match="policy invariants differ"):
        MODULE._write_sealed_unique(tmp_path / "tampered.json", tampered)


def test_generated_output_names_are_unique_and_self_announcing(tmp_path):
    first = MODULE._unique_output_path(tmp_path, "results")
    second = MODULE._unique_output_path(tmp_path, "results")
    assert first != second
    assert MODULE.SAFE_RESULT_RE.fullmatch(first.name)
    assert "Qwen3-30B-A3B-Instruct-2507" in first.name
