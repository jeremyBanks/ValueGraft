from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import powered_v13_technical as technical
from powered_v13_schema import MODEL_ID, MODEL_REVISION, R2
from powered_v13_tokens import (
    build_probe_generation_prefix,
    probe_target_ids,
)


EXPECTED = {
    "technical_e01": {
        "stable_technical_id": (
            "9aa6330b023ae94bef2f34cc8cb480adc21aff6381d2af03cd74a11862c22bcb"
        ),
        "technical_input_sha256": (
            "d88fb910c5e9d4c94a099205fe50052ad3aa13a8ef4fc3211287b25eaf562ca1"
        ),
        "technical_probe_sha256": (
            "c53015f504888fef31be1c2439a3f06c1c06fee931987b69bfd00c992e83dcde"
        ),
        "technical_case_sha256": (
            "e1bf262877a6b0eceda06300991e2f3a8f1213631e342dedd50412d86862112a"
        ),
        "source_token_count": 1272,
        "source_geometry_sha256": (
            "0d8e380ae41a1c745eb2e7c0ff88b9245c9476a3d9874173c6eb7d24417cf518"
        ),
        "fresh_geometry_sha256": (
            "395459c7b17244cbe12a257ed89dd38b66b14043490d3d484b57a5b26b34f6e1"
        ),
    },
    "technical_long": {
        "stable_technical_id": (
            "c91600bdf6071c1b59f0eae203b2d7b5d1aa374ec3c046d2430299c826e91f13"
        ),
        "technical_input_sha256": (
            "7587f53db894a5053f5a14da30c4c919c518b67c14e55ac90379a169c3c00411"
        ),
        "technical_probe_sha256": (
            "b1f3dd890014368b61fadf4eaaaff23f64636775aee46e288dd5db225f86ebb7"
        ),
        "technical_case_sha256": (
            "b8a23715656a63beb6c23a251a4c5f5e816246de998371928d89e8ce84610d37"
        ),
        "source_token_count": 4535,
        "source_geometry_sha256": (
            "a733b1923ce6f2a1cc81f4780fc84d985fcbfe447658d4ef499b46c19c9ce60a"
        ),
        "fresh_geometry_sha256": (
            "81727735a438dae7ae3201fae746413a61dc86a309733df7557e395f2992013a"
        ),
    },
}


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned production tokenizer is not cached")


def test_fixed_e01_bytes_and_only_required_history_fields_are_loaded() -> None:
    result = technical.load_fixed_e01(Path("."))
    assert result["source_path"] == technical.E01_RELATIVE_PATH
    assert result["source_sha256"] == technical.E01_SHA256
    assert result["middle_end_msg"] == 15
    assert len(result["correct"]) == len(result["wrong"]) == 17
    assert result["correct"][15:] == result["wrong"][15:]
    assert result["correct"] != result["wrong"]


@pytest.mark.parametrize("case_id", technical.TECHNICAL_CASE_IDS)
def test_exact_technical_plan_evidence_is_stable_and_semantic_n_zero(
    tokenizer, case_id: str,
) -> None:
    result = technical.build_technical_case(tokenizer, Path("."), case_id)
    expected = EXPECTED[case_id]
    assert result["schema"] == technical.TECHNICAL_INPUT_SCHEMA
    assert result["case_id"] == case_id
    assert result["stable_technical_id"] == expected["stable_technical_id"]
    assert result["technical_input_sha256"] == \
        expected["technical_input_sha256"]
    assert result["semantic_n"] == 0
    assert result["production_pool_imported"] is False
    assert result["production_entropy_requested"] is False
    assert result["production_probe_reachable"] is False
    assert result["score_values_present"] is False
    assert result["technical_probe_sha256"] == \
        expected["technical_probe_sha256"]
    assert result["technical_case_sha256"] == \
        expected["technical_case_sha256"]
    assert result["technical_probe"]["semantic_n"] == 0
    assert result["technical_probe"]["inferential_use"] == "FORBIDDEN"
    probe_core = {key: value for key, value in result["technical_probe"].items()
                  if key != "technical_probe_sha256"}
    assert technical.sha256_json(probe_core) == \
        result["technical_probe_sha256"]
    assert result["selected_ids_and_positions_identical"] is True
    assert result["plans"]["C"]["token_count"] == \
        result["plans"]["W"]["token_count"] == \
        expected["source_token_count"]
    assert result["plans"]["C"]["geometry_sha256"] == \
        result["plans"]["W"]["geometry_sha256"] == \
        expected["source_geometry_sha256"]
    assert result["plans"]["F"]["geometry_sha256"] == \
        expected["fresh_geometry_sha256"]
    for row in result["plans"].values():
        assert row["r2_width"] == 74
        assert row["r2_end"] - row["r2_start"] == 74
        assert row["max_call_width"] <= 4096
    assert len({tuple(row["r2_token_ids"])
                for row in result["plans"].values()}) == 1
    assert len({tuple(row["r2_logical_positions"])
                for row in result["plans"].values()}) == 1
    core = {key: value for key, value in result.items()
            if key not in {
                "technical_input_sha256", "technical_probe",
                "technical_case_sha256", "plan_objects", "context_messages",
            }}
    assert result["technical_input_sha256"] == technical.sha256_json(core)
    assert result["technical_case_sha256"] == technical.sha256_json({
        "technical_input_sha256": result["technical_input_sha256"],
        "technical_probe_sha256": result["technical_probe_sha256"],
    })
    assert result["plan_objects"]["C"].regions.interval(R2)[1] - \
        result["plan_objects"]["C"].regions.interval(R2)[0] == 74

    probe = result["technical_probe"]
    for history in ("C", "W", "F"):
        prefix = build_probe_generation_prefix(
            tokenizer, result["context_messages"][history], probe["probe"])
        plan = result["plan_objects"][history]
        assert prefix[:len(plan.token_ids)] == list(plan.token_ids)
        for target in (
            probe["correct_target"], probe["counterfactual_target"],
        ):
            assert probe_target_ids(
                tokenizer, result["context_messages"][history],
                probe["probe"], target)


def test_long_case_is_literal_4_5k_geometry_with_same_compacted_text(tokenizer) -> None:
    short = technical.build_technical_case(
        tokenizer, Path("."), "technical_e01")
    long = technical.build_technical_case(
        tokenizer, Path("."), "technical_long")
    low, high = technical.TECHNICAL_LONG_SOURCE_TOKEN_BOUNDS
    assert low <= long["plans"]["C"]["token_count"] <= high
    assert long["plans"]["C"]["max_call_width"] == 3260
    assert long["plans"]["F"]["token_ids_sha256"] == \
        short["plans"]["F"]["token_ids_sha256"]
    assert long["plans"]["F"]["geometry_sha256"] != \
        short["plans"]["F"]["geometry_sha256"]
    assert long["plans"]["F"]["r2_logical_positions"] != \
        short["plans"]["F"]["r2_logical_positions"]
    assert long["context_messages"]["F"] == short["context_messages"]["F"]
    fresh_text = "\n".join(
        message["content"] for message in long["context_messages"]["F"])
    assert technical.TECHNICAL_LONG_TARGET in fresh_text
    assert technical.TECHNICAL_LONG_COUNTERTARGET in fresh_text
    assert technical.TECHNICAL_LONG_FILLER_UNIT.strip() not in fresh_text


def test_e01_corruption_and_symlink_fail_before_planning(
    tmp_path: Path, tokenizer,
) -> None:
    destination = tmp_path / technical.E01_RELATIVE_PATH
    destination.parent.mkdir(parents=True)
    original = Path(technical.E01_RELATIVE_PATH).read_bytes()
    destination.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(technical.V13TechnicalError, match="byte hash differs"):
        technical.build_technical_case(
            tokenizer, tmp_path, "technical_e01")
    destination.unlink()
    destination.symlink_to(Path(technical.E01_RELATIVE_PATH).resolve())
    with pytest.raises(technical.V13TechnicalError, match="symlinked"):
        technical.load_fixed_e01(tmp_path)


def test_unknown_case_and_forbidden_import_surface_fail_closed(tokenizer) -> None:
    with pytest.raises(technical.V13TechnicalError,
                       match="unknown technical case"):
        technical.build_technical_case(tokenizer, Path("."), "e01")
    source = inspect.getsource(technical)
    for forbidden in (
        "from powered_v13_recipe",
        "import powered_v13_recipe",
        "from powered_v13_permutation",
        "import powered_v13_permutation",
        "from powered_v13_stimuli",
        "import powered_v13_stimuli",
    ):
        assert forbidden not in source


def test_literal_probe_hashes_and_e01_source_contract_fail_closed(
        monkeypatch) -> None:
    monkeypatch.setattr(technical, "E01_PROBE", technical.E01_PROBE + " drift")
    with pytest.raises(technical.V13TechnicalError,
                       match="probe literal hash differs"):
        technical._technical_probe_contract("technical_e01")
