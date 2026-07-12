from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_legacy_head_map_lineage.py"
SPEC = importlib.util.spec_from_file_location("analyze_legacy_head_map_lineage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_committed_head_map_lineage_discrepancy() -> None:
    report = MODULE.build_report(REPO_ROOT, "2026-07-12T00:00:00Z")
    comparison = report["head_map_comparison"]
    assert comparison["evaluated_slot_count"] == 109
    assert comparison["tracked_positive_slot_count"] == 121
    assert comparison["overlap_count"] == 80
    assert comparison["evaluated_but_not_tracked_positive_count"] == 29
    assert comparison["tracked_positive_but_not_evaluated_count"] == 41
    assert comparison["nonempty_profile_subset_count_tested"] == 1023
    assert comparison["matching_subset_count"] == 0
    assert comparison["derivation_reproduced"] is False


def test_embedded_combination_maps_reconstruct() -> None:
    report = MODULE.build_report(REPO_ROOT, "2026-07-12T00:00:00Z")
    checks = report["combination_checks"]
    assert checks["tracked_layer_map_equals_evaluated_layer_map"] is True
    assert checks["selected_layer_count"] == 27
    assert checks["intersection_reconstructs_from_evaluated_heads_and_layers"] is True
    assert checks["union_reconstructs_from_evaluated_heads_and_layers"] is True
    assert checks["reconstructed_intersection_slot_count"] == 74
    assert checks["reconstructed_union_slot_count"] == 143
