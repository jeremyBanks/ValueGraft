from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_coherent_canary_v12_harvest.py"
SPEC = importlib.util.spec_from_file_location(
    "aggregate_coherent_canary_v12_harvest_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def runtime(subject="exact-subject") -> dict:
    expected = MODULE.SUBJECTS[subject]
    return {
        "subject": subject,
        "requested_model": expected["requested_model"],
        "requested_revision": expected["requested_revision"],
        "dtype": "torch.bfloat16",
        "attention_backend": "eager",
        "fingerprint_sha256": ("a" if subject == "exact-subject" else "b") * 64,
    }


def case_report(case_id: str, *, subject="exact-subject",
                full=(1.0, 0.8, 0.7, 0.9),
                value=(0.6, 0.5, 0.4, 0.55)) -> dict:
    # Tuple fields are D_N, SEL_N, Hplus_N, D_P.
    families = {}
    yardsticks = {}
    for name, values in (("full_KV", full), ("value_only", value)):
        d_n, sel_n, hplus_n, d_p = values
        families[name] = {
            "N": {
                "D_focal": d_n, "SEL": sel_n, "Hplus": hplus_n,
                "D_nonfocal": 0.0, "U": 0.0, "Uplus": 0.0,
            },
            "P": {
                "D_focal": d_p, "SEL": sel_n, "Hplus": hplus_n,
                "D_nonfocal": 0.0, "U": 0.0, "Uplus": 0.0,
            },
        }
        yardsticks[name] = {
            "D_focal_N": d_n,
            "D_focal_P": d_p,
            "absolute_N_minus_P": abs(d_n - d_p),
            "three_times_absolute_N_minus_P": 3 * abs(d_n - d_p),
        }
    return {
        "schema": MODULE.CASE_REPORT_SCHEMA,
        "design_id": MODULE.DESIGN_ID,
        "case_id": case_id,
        "subject": subject,
        "runtime": runtime(subject),
        "semantic_evidence_eligible": subject == "exact-subject",
        "apparatus_integration_only": subject == "local-apparatus",
        "treatment_artifact": {"path": f"treatment-{case_id}.json",
                               "sha256": "c" * 64},
        "R2_primary_estimands": {
            "N": {family: rows["N"] for family, rows in families.items()},
            "P": {family: rows["P"] for family, rows in families.items()},
        },
        "schedule_yardstick_components": yardsticks,
        "phase_a_fresh_damage_diagnostic": {
            "margin": 0.8,
            "correct_target_logprob": 0.7,
            "positive_margin_damage": True,
            "positive_correct_target_damage": True,
        },
    }


def reports(tmp_path: Path, rows: list[dict]) -> list[Path]:
    return [write_json(tmp_path / f"{row['case_id']}-{index}.json", row)
            for index, row in enumerate(rows)]


def test_one_case_is_descriptive_incomplete_with_full_statistics(tmp_path):
    paths = reports(tmp_path, [case_report("e01")])
    aggregate = MODULE.aggregate(paths, repo_root=tmp_path)
    assert aggregate["stage"] == "ONE_CASE_DESCRIPTIVE"
    assert aggregate["status"] == "INCOMPLETE"
    assert aggregate["semantic_decisions_issued"] is False
    assert aggregate["families"]["full_KV"]["classification"] == "INCOMPLETE"
    summary = aggregate["families"]["full_KV"]["summaries"]["D_N"]
    assert summary == {
        "n": 1, "values_by_case": {"e01": 1.0},
        "mean": 1.0, "median": 1.0, "sample_sd": None,
        "sign_count": {"positive": 1, "zero": 0, "negative": 0},
    }
    assert aggregate["inference"]["p_values_computed"] is False
    assert aggregate["input_reports"][0]["sha256"] == MODULE.file_sha256(paths[0])


def test_local_case_can_only_be_apparatus_integration_incomplete(tmp_path):
    paths = reports(tmp_path, [case_report("e01", subject="local-apparatus")])
    aggregate = MODULE.aggregate(paths, repo_root=tmp_path)
    assert aggregate["status"] == "INCOMPLETE"
    assert aggregate["semantic_decisions_issued"] is False
    assert aggregate["semantic_evidence_eligible"] is False
    assert aggregate["apparatus_integration_only"] is True

    local_four = [case_report(case_id, subject="local-apparatus")
                  for case_id in MODULE.PRIMARY_CASE_IDS]
    aggregate = MODULE.aggregate(
        reports(tmp_path / "four", local_four), repo_root=tmp_path)
    assert aggregate["stage"] == "LOCAL_DESCRIPTIVE"
    assert aggregate["semantic_decisions_issued"] is False
    assert {row["classification"] for row in aggregate["families"].values()} == {
        "INCOMPLETE"}


def pass_four_rows() -> list[dict]:
    full = [
        (1.0, 0.8, 0.6, 0.9),
        (0.9, 0.7, 0.5, 0.8),
        (0.8, 0.6, 0.4, 0.7),
        (-0.1, -0.2, 0.3, -0.05),
    ]
    value = [
        (0.6, 0.5, 0.4, 0.55),
        (0.5, 0.4, 0.35, 0.45),
        (0.4, 0.3, 0.3, 0.35),
        (-0.05, -0.1, 0.2, -0.02),
    ]
    return [case_report(case_id, full=full[index], value=value[index])
            for index, case_id in enumerate(MODULE.PRIMARY_CASE_IDS)]


def test_four_case_pass4_and_direct_paired_differences(tmp_path):
    paths = reports(tmp_path, pass_four_rows())
    aggregate = MODULE.aggregate(paths, repo_root=tmp_path)
    assert aggregate["stage"] == "FOUR_CASE_PRIMARY"
    assert aggregate["semantic_decisions_issued"] is True
    assert aggregate["families"]["full_KV"]["classification"] == "PASS4"
    assert aggregate["families"]["value_only"]["classification"] == "PASS4"
    summary = aggregate["families"]["full_KV"]["summaries"]["D_N"]
    expected = [1.0, 0.9, 0.8, -0.1]
    assert summary["mean"] == pytest.approx(statistics.fmean(expected))
    assert summary["median"] == pytest.approx(statistics.median(expected))
    assert summary["sample_sd"] == pytest.approx(statistics.stdev(expected))
    assert summary["sign_count"] == {"positive": 3, "zero": 0, "negative": 1}
    paired = aggregate["paired_family_differences"]
    assert paired["contrast"] == "full_KV_minus_value_only"
    assert paired["per_case"]["e01"]["D_N"] == pytest.approx(0.4)
    assert paired["inferential_test_performed"] is False
    assert aggregate["families"]["full_KV"]["summaries"]["U_N"]["n"] == 4
    assert aggregate["phase_a_utility_damage"]["e01"][
        "utility_damage_eligible"] is True


def test_four_case_stop4_and_ambiguous4_are_family_specific(tmp_path):
    rows = pass_four_rows()
    # Full KV has only one directionally selective positive case -> STOP4.
    full = [
        (0.8, 0.4, 0.5, 0.75),
        (0.7, -0.2, 0.5, 0.65),
        (0.6, -0.1, 0.5, 0.55),
        (-0.1, -0.2, 0.5, -0.05),
    ]
    # Value-only has two directionally selective cases and positive means:
    # neither PASS4 nor STOP4, hence AMBIGUOUS4.
    value = [
        (0.6, 0.4, 0.4, 0.55),
        (0.5, 0.3, 0.4, 0.45),
        (0.4, -0.1, 0.4, 0.35),
        (0.3, -0.1, 0.4, 0.25),
    ]
    for index, row in enumerate(rows):
        replacement = case_report(
            row["case_id"], full=full[index], value=value[index])
        rows[index] = replacement
    aggregate = MODULE.aggregate(reports(tmp_path, rows), repo_root=tmp_path)
    assert aggregate["families"]["full_KV"]["classification"] == "STOP4"
    assert aggregate["families"]["value_only"]["classification"] == "AMBIGUOUS4"


def test_duplicate_case_or_runtime_difference_is_rejected(tmp_path):
    rows = pass_four_rows()
    rows[-1]["case_id"] = "e03"
    with pytest.raises(MODULE.AggregateError, match="IDs are not unique"):
        MODULE.aggregate(reports(tmp_path, rows), repo_root=tmp_path)

    rows = pass_four_rows()
    rows[-1]["runtime"] = runtime("local-apparatus")
    rows[-1]["subject"] = "local-apparatus"
    rows[-1]["semantic_evidence_eligible"] = False
    rows[-1]["apparatus_integration_only"] = True
    with pytest.raises(MODULE.AggregateError, match="identical subject/runtime"):
        MODULE.aggregate(reports(tmp_path / "runtime", rows), repo_root=tmp_path)


def make_ambiguous_four(tmp_path: Path) -> tuple[list[dict], list[Path], Path, dict]:
    rows = pass_four_rows()
    # Preserve full_KV PASS4. Make value_only AMBIGUOUS4 with two selective cases.
    value = [
        (0.6, 0.4, 0.4, 0.55),
        (0.5, 0.3, 0.4, 0.45),
        (0.4, -0.1, 0.4, 0.35),
        (0.3, -0.1, 0.4, 0.25),
    ]
    for index, row in enumerate(rows):
        full = (
            row["R2_primary_estimands"]["N"]["full_KV"]["D_focal"],
            row["R2_primary_estimands"]["N"]["full_KV"]["SEL"],
            row["R2_primary_estimands"]["N"]["full_KV"]["Hplus"],
            row["R2_primary_estimands"]["P"]["full_KV"]["D_focal"],
        )
        rows[index] = case_report(
            row["case_id"], full=full, value=value[index])
    paths = reports(tmp_path, rows)
    four = MODULE.aggregate(paths, repo_root=tmp_path)
    output = write_json(tmp_path / "aggregate-four.json", four)
    return rows, paths, output, four


def test_six_case_only_reclassifies_ambiguous_family(tmp_path):
    rows, primary_paths, four_path, four = make_ambiguous_four(tmp_path)
    assert four["families"]["full_KV"]["classification"] == "PASS4"
    assert four["families"]["value_only"]["classification"] == "AMBIGUOUS4"
    reserve = [
        case_report("e05", full=(-2.0, -1.0, -0.5, 2.0),
                    value=(0.7, 0.5, 0.4, 0.65)),
        case_report("e06", full=(-2.0, -1.0, -0.5, 2.0),
                    value=(0.8, 0.6, 0.4, 0.75)),
    ]
    reserve_paths = reports(tmp_path / "reserve", reserve)
    aggregate = MODULE.aggregate(
        primary_paths + reserve_paths, repo_root=tmp_path,
        four_case_aggregate=four_path)
    assert aggregate["stage"] == "SIX_CASE_RESERVE"
    assert aggregate["families"]["full_KV"]["classification"] == "PASS4"
    assert aggregate["families"]["full_KV"]["reserve_role"].startswith(
        "descriptive_only")
    assert aggregate["families"]["full_KV"]["reserve_descriptive"][
        "semantic_reclassification_performed"] is False
    assert aggregate["families"]["full_KV"]["reserve_descriptive"][
        "case_ids"] == ["e05", "e06"]
    assert aggregate["families"]["value_only"]["classification"] == "PASS6"
    assert aggregate["families"]["value_only"]["reserve_descriptive"][
        "semantic_reclassification_performed"] is True
    assert aggregate["four_case_aggregate"]["sha256"] == MODULE.file_sha256(
        four_path)


def test_ambiguous_family_that_misses_six_case_rule_becomes_stop6(tmp_path):
    _, primary_paths, four_path, _ = make_ambiguous_four(tmp_path)
    reserve = [
        case_report("e05", value=(-0.8, -0.4, -0.2, -0.75)),
        case_report("e06", value=(-0.9, -0.5, -0.2, -0.85)),
    ]
    aggregate = MODULE.aggregate(
        primary_paths + reports(tmp_path / "reserve", reserve),
        repo_root=tmp_path, four_case_aggregate=four_path)
    assert aggregate["families"]["value_only"]["classification"] == "STOP6"


def test_six_requires_bound_four_case_report_and_literal_primary_hashes(tmp_path):
    _, primary_paths, four_path, _ = make_ambiguous_four(tmp_path)
    reserve_paths = reports(tmp_path / "reserve", [
        case_report("e05"), case_report("e06")])
    with pytest.raises(MODULE.AggregateError, match="requires a bound four-case"):
        MODULE.aggregate(primary_paths + reserve_paths, repo_root=tmp_path)

    changed = json.loads(primary_paths[0].read_text())
    changed["completed_at_utc"] = "changed-after-four-case-aggregate"
    write_json(primary_paths[0], changed)
    with pytest.raises(MODULE.AggregateError, match="does not bind current"):
        MODULE.aggregate(
            primary_paths + reserve_paths, repo_root=tmp_path,
            four_case_aggregate=four_path)


def test_six_is_rejected_when_four_case_result_has_no_ambiguity(tmp_path):
    primary_paths = reports(tmp_path / "primary", pass_four_rows())
    four = MODULE.aggregate(primary_paths, repo_root=tmp_path)
    assert {row["classification"] for row in four["families"].values()} == {
        "PASS4"}
    four_path = write_json(tmp_path / "aggregate-four.json", four)
    reserve_paths = reports(tmp_path / "reserve", [
        case_report("e05"), case_report("e06")])
    with pytest.raises(MODULE.AggregateError, match="not authorized"):
        MODULE.aggregate(
            primary_paths + reserve_paths, repo_root=tmp_path,
            four_case_aggregate=four_path)


def test_yardstick_inconsistency_nonfinite_and_create_only_are_rejected(tmp_path):
    row = case_report("e01")
    row["schedule_yardstick_components"]["full_KV"][
        "absolute_N_minus_P"] = 99.0
    path = reports(tmp_path, [row])[0]
    with pytest.raises(MODULE.AggregateError, match="yardstick differs"):
        MODULE.aggregate([path], repo_root=tmp_path)

    row = case_report("e01")
    row["R2_primary_estimands"]["N"]["full_KV"]["D_focal"] = float("nan")
    path = reports(tmp_path / "nan", [row])[0]
    with pytest.raises(MODULE.AggregateError, match="nonfinite"):
        MODULE.aggregate([path], repo_root=tmp_path)

    good = MODULE.aggregate(
        reports(tmp_path / "good", [case_report("e01")]), repo_root=tmp_path)
    output = tmp_path / "aggregate.json"
    MODULE.write_exclusive_json(output, good)
    with pytest.raises(FileExistsError):
        MODULE.write_exclusive_json(output, good)


def test_module_imports_no_model_or_tokenizer_code():
    tree = ast.parse(SCRIPT.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    assert not any(name.startswith(("torch", "transformers", "coherent_canary"))
                   for name in imports)
