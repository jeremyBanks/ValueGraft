from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_coherent_canary_v12_e01_tokens import decompose, reconstruct  # noqa: E402


PACKAGE = ROOT / (
    "results/coherent_canary_v12_treatment_package/"
    "coherent-canary-v12-treatment-e01_exact-subject_20260712T040237879370Z"
)


def test_committed_e01_token_decomposition() -> None:
    manifest, raw = reconstruct(PACKAGE)
    assert manifest["original"]["sha256"] == (
        "f6622d978c80d8f4f874c208ef9f6d9546ab3f1bf7676655c9a0b4662718d082"
    )
    result = decompose(raw)
    assert result["primary_arm_count"] == 31
    assert result["available_placebo_control_count"] == 0
    assert result["unique_primary_focal_generations"] == ["Ring 3"]
    assert result["unique_primary_nonfocal_generations"] == ["30 days"]

    rows = {(row["schedule"], row["family"]): row
            for row in result["decompositions"]}
    n_value = rows[("N", "value_only")]
    p_value = rows[("P", "value_only")]
    assert n_value["semantic_margin_token_differences"] == [
        0.15625, 0.192840576171875
    ]
    assert n_value["first_token_semantic_direction_positive"] is True
    assert n_value["all_token_semantic_directions_positive"] is True
    assert p_value["semantic_margin_token_differences"] == [
        -0.1875, 0.22959303855895996
    ]
    assert p_value["first_token_semantic_direction_positive"] is False
    assert p_value["all_token_semantic_directions_positive"] is False
