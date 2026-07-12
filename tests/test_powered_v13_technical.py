from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import powered_v13_technical as technical
from powered_v13_schema import MODEL_ID, MODEL_REVISION, R2


EXPECTED = {
    "technical_e01": {
        "stable_technical_id": (
            "512d7a10fce458d426914f77aa36408b7e2cd6c65c99bad2e6ca26a75b478f0e"
        ),
        "technical_input_sha256": (
            "075811985d5dc13c5c92528ddcf0cf6cbf12d56bf3e0ece81ca6e834826b3b49"
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
            "dad43612b46946b81dbd123dfb328484143e35ec239b40db111abeb47030b170"
        ),
        "technical_input_sha256": (
            "692f7e5436fa24198e5850734a1cb0ffd51db60bf287913ed9c4a0507d8e58aa"
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
    assert result["probe_or_score_present"] is False
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
            if key not in {"technical_input_sha256", "plan_objects"}}
    assert result["technical_input_sha256"] == technical.sha256_json(core)
    assert result["plan_objects"]["C"].regions.interval(R2)[1] - \
        result["plan_objects"]["C"].regions.interval(R2)[0] == 74


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
