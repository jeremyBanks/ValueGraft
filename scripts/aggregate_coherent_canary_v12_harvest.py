#!/usr/bin/env python3
"""Apply the frozen v12 four-case and six-case aggregate stopping rules.

The input is one report per independently harvested treatment case.  This
script performs no model forward, tokenizer reconstruction, or p-value test. It
binds the literal report bytes, checks their shared subject/runtime, summarizes
the fixed-case contrasts, and applies only the preregistered stopping rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DESIGN_ID = "coherent-state-decision-canary-v12"
CASE_REPORT_SCHEMA = "coherent_state_decision_canary_v12_treatment_harvest_v1"
AGGREGATE_SCHEMA = "coherent_state_decision_canary_v12_treatment_aggregate_v1"
FAMILIES = ("full_KV", "value_only")
PRIMARY_CASE_IDS = ("e01", "e02", "e03", "e04")
RESERVE_CASE_IDS = ("e05", "e06")
ALL_CASE_IDS = PRIMARY_CASE_IDS + RESERVE_CASE_IDS
SUBJECTS = {
    "exact-subject": {
        "requested_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "requested_revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    },
    "local-apparatus": {
        "requested_model": "Qwen/Qwen3-0.6B",
        "requested_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    },
}


class AggregateError(RuntimeError):
    """A case report, aggregate binding, or stopping-rule input differs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as exc:
        raise AggregateError(f"cannot parse {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{label} is nonfinite")
    return result


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def summarize(values_by_case: Mapping[str, float]) -> dict[str, Any]:
    ordered = {case_id: float(values_by_case[case_id])
               for case_id in sorted(values_by_case)}
    values = list(ordered.values())
    require(bool(values), "cannot summarize an empty case set")
    return {
        "n": len(values),
        "values_by_case": ordered,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "sign_count": {
            "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values),
            "negative": sum(value < 0 for value in values),
        },
    }


def validate_runtime(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    runtime = report.get("runtime")
    require(isinstance(runtime, Mapping), f"{label} runtime is absent")
    subject = runtime.get("subject")
    require(subject in SUBJECTS, f"{label} subject differs")
    expected = SUBJECTS[str(subject)]
    require(runtime.get("requested_model") == expected["requested_model"] and
            runtime.get("requested_revision") == expected["requested_revision"] and
            runtime.get("dtype") == "torch.bfloat16" and
            runtime.get("attention_backend") == "eager" and
            is_sha256(runtime.get("fingerprint_sha256")),
            f"{label} pinned runtime differs")
    require(report.get("subject") == subject and
            report.get("semantic_evidence_eligible") is
            (subject == "exact-subject") and
            report.get("apparatus_integration_only") is
            (subject == "local-apparatus"),
            f"{label} subject/semantic/apparatus labels differ")
    return dict(runtime)


def extract_case(path: Path, repo_root: Path) -> dict[str, Any]:
    report = load_object(path)
    case_id = report.get("case_id")
    require(report.get("schema") == CASE_REPORT_SCHEMA and
            report.get("design_id") == DESIGN_ID and
            case_id in ALL_CASE_IDS,
            f"case report schema/design/ID differs: {path}")
    treatment_artifact = report.get("treatment_artifact")
    require(isinstance(treatment_artifact, Mapping) and
            isinstance(treatment_artifact.get("path"), str) and
            is_sha256(treatment_artifact.get("sha256")),
            f"case {case_id} treatment-artifact binding differs")
    runtime = validate_runtime(report, f"case {case_id}")
    estimands = report.get("R2_primary_estimands")
    yardsticks = report.get("schedule_yardstick_components")
    require(isinstance(estimands, Mapping) and set(estimands) == {"N", "P"} and
            all(isinstance(estimands[schedule], Mapping) and
                set(estimands[schedule]) == set(FAMILIES)
                for schedule in ("N", "P")),
            f"case {case_id} R2 estimand coverage differs")
    require(isinstance(yardsticks, Mapping) and
            set(yardsticks) == set(FAMILIES),
            f"case {case_id} schedule-yardstick coverage differs")

    family_values: dict[str, dict[str, float]] = {}
    for family in FAMILIES:
        n_row = estimands["N"][family]
        p_row = estimands["P"][family]
        yardstick = yardsticks[family]
        require(isinstance(n_row, Mapping) and isinstance(p_row, Mapping) and
                isinstance(yardstick, Mapping),
                f"case {case_id} {family} records differ")
        d_n = finite_number(n_row.get("D_focal"), f"{case_id}.{family}.D_N")
        d_nonfocal_n = finite_number(
            n_row.get("D_nonfocal"), f"{case_id}.{family}.D_nonfocal_N")
        sel_n = finite_number(n_row.get("SEL"), f"{case_id}.{family}.SEL_N")
        hplus_n = finite_number(
            n_row.get("Hplus"), f"{case_id}.{family}.Hplus_N")
        utility_n = finite_number(n_row.get("U"), f"{case_id}.{family}.U_N")
        utility_plus_n = finite_number(
            n_row.get("Uplus"), f"{case_id}.{family}.Uplus_N")
        d_p = finite_number(p_row.get("D_focal"), f"{case_id}.{family}.D_P")
        absolute_gap = abs(d_n - d_p)
        declared = {
            "D_N": finite_number(
                yardstick.get("D_focal_N"), f"{case_id}.{family}.yardstick.D_N"),
            "D_P": finite_number(
                yardstick.get("D_focal_P"), f"{case_id}.{family}.yardstick.D_P"),
            "absolute_gap": finite_number(
                yardstick.get("absolute_N_minus_P"),
                f"{case_id}.{family}.yardstick.absolute_gap"),
        }
        require(close(declared["D_N"], d_n) and
                close(declared["D_P"], d_p) and
                close(declared["absolute_gap"], absolute_gap),
                f"case {case_id} {family} yardstick differs from estimands")
        family_values[family] = {
            "D_N": d_n,
            "D_nonfocal_N": d_nonfocal_n,
            "SEL_N": sel_n,
            "Hplus_N": hplus_n,
            "U_N": utility_n,
            "Uplus_N": utility_plus_n,
            "D_P": d_p,
            "absolute_schedule_gap": absolute_gap,
        }
    damage = report.get("phase_a_fresh_damage_diagnostic")
    require(isinstance(damage, Mapping),
            f"case {case_id} Phase-A damage diagnostic is absent")
    phase_damage = {
        "margin": finite_number(
            damage.get("margin"), f"{case_id}.phase_a_damage.margin"),
        "correct_target_logprob": finite_number(
            damage.get("correct_target_logprob"),
            f"{case_id}.phase_a_damage.correct_target_logprob"),
    }
    phase_damage["utility_damage_eligible"] = (
        phase_damage["margin"] > 0 and
        phase_damage["correct_target_logprob"] > 0)
    return {
        "case_id": str(case_id),
        "path": path,
        "sha256": file_sha256(path),
        "display_path": display_path(path, repo_root),
        "runtime": runtime,
        "families": family_values,
        "phase_a_fresh_damage": phase_damage,
    }


def family_analysis(cases: Sequence[Mapping[str, Any]], family: str,
                    *, directional_threshold: int) -> dict[str, Any]:
    per_case = {
        str(case["case_id"]): dict(case["families"][family]) for case in cases}
    summaries = {
        metric: summarize({case_id: values[metric]
                           for case_id, values in per_case.items()})
        for metric in (
            "D_N", "D_nonfocal_N", "SEL_N", "Hplus_N", "U_N",
            "Uplus_N", "D_P", "absolute_schedule_gap")
    }
    directional_selective = sum(
        values["D_N"] > 0 and values["SEL_N"] > 0
        for values in per_case.values())
    mean_d = summaries["D_N"]["mean"]
    mean_hplus = summaries["Hplus_N"]["mean"]
    mean_gap = summaries["absolute_schedule_gap"]["mean"]
    yardstick_pass = mean_d >= 3.0 * mean_gap
    return {
        "per_case": per_case,
        "summaries": summaries,
        "criteria": {
            "mean_D_N_positive": mean_d > 0,
            "directional_and_selective_count": directional_selective,
            "directional_and_selective_threshold": directional_threshold,
            "directional_and_selective_threshold_met":
                directional_selective >= directional_threshold,
            "mean_Hplus_N_positive": mean_hplus > 0,
            "schedule_yardstick_met": yardstick_pass,
            "schedule_yardstick_mean_D_N": mean_d,
            "schedule_yardstick_three_times_mean_absolute_gap": 3.0 * mean_gap,
        },
    }


def classify_four(analysis: Mapping[str, Any]) -> str:
    criteria = analysis["criteria"]
    if (criteria["mean_D_N_positive"] and
            criteria["directional_and_selective_threshold_met"] and
            criteria["mean_Hplus_N_positive"] and
            criteria["schedule_yardstick_met"]):
        return "PASS4"
    if (not criteria["mean_D_N_positive"] or
            criteria["directional_and_selective_count"] <= 1):
        return "STOP4"
    return "AMBIGUOUS4"


def classify_six(analysis: Mapping[str, Any]) -> str:
    criteria = analysis["criteria"]
    if (criteria["mean_D_N_positive"] and
            criteria["directional_and_selective_threshold_met"] and
            criteria["mean_Hplus_N_positive"] and
            criteria["schedule_yardstick_met"]):
        return "PASS6"
    return "STOP6"


def paired_family_differences(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "D_N", "D_nonfocal_N", "SEL_N", "Hplus_N", "U_N", "Uplus_N",
        "D_P", "absolute_schedule_gap")
    per_case = {}
    for case in cases:
        full = case["families"]["full_KV"]
        value = case["families"]["value_only"]
        per_case[str(case["case_id"])] = {
            metric: full[metric] - value[metric] for metric in metrics}
    return {
        "contrast": "full_KV_minus_value_only",
        "per_case": per_case,
        "summaries": {
            metric: summarize({case_id: values[metric]
                               for case_id, values in per_case.items()})
            for metric in metrics
        },
        "inferential_test_performed": False,
    }


def validate_four_case_aggregate(path: Path, *, repo_root: Path,
                                 cases: Sequence[Mapping[str, Any]],
                                 runtime: Mapping[str, Any]) -> dict[str, Any]:
    report = load_object(path)
    exact = runtime.get("subject") == "exact-subject"
    expected_stage = "FOUR_CASE_PRIMARY" if exact else "LOCAL_DESCRIPTIVE"
    require(report.get("schema") == AGGREGATE_SCHEMA and
            report.get("design_id") == DESIGN_ID and
            report.get("stage") == expected_stage and
            report.get("case_ids") == list(PRIMARY_CASE_IDS) and
            report.get("subject") == runtime.get("subject") and
            report.get("semantic_decisions_issued") is exact and
            report.get("runtime") == runtime,
            "bound four-case aggregate identity/runtime differs")
    families = report.get("families")
    allowed = ({"PASS4", "STOP4", "AMBIGUOUS4"}
               if exact else {"INCOMPLETE"})
    require(isinstance(families, Mapping) and set(families) == set(FAMILIES) and
            all(families[family].get("classification") in allowed
                for family in FAMILIES),
            "bound four-case family classifications differ")
    prior_inputs = report.get("input_reports")
    require(isinstance(prior_inputs, list) and len(prior_inputs) == 4,
            "bound four-case input bindings differ")
    prior_hashes = {
        row.get("case_id"): row.get("sha256") for row in prior_inputs
        if isinstance(row, Mapping)
    }
    current_hashes = {case["case_id"]: case["sha256"] for case in cases
                      if case["case_id"] in PRIMARY_CASE_IDS}
    require(set(prior_hashes) == set(PRIMARY_CASE_IDS) and
            all(is_sha256(value) for value in prior_hashes.values()) and
            prior_hashes == current_hashes,
            "bound four-case aggregate does not bind current primary reports")
    primary_cases = [case for case in cases if case["case_id"] in PRIMARY_CASE_IDS]
    for family in FAMILIES:
        expected_classification = (
            classify_four(family_analysis(
                primary_cases, family, directional_threshold=3))
            if exact else "INCOMPLETE")
        require(families[family].get("classification") == expected_classification,
                f"bound four-case {family} classification differs from inputs")
    return {
        "path": display_path(path, repo_root),
        "sha256": file_sha256(path),
        "report": report,
    }


def aggregate(report_paths: Sequence[Path], *, repo_root: Path = ROOT,
              four_case_aggregate: Path | None = None) -> dict[str, Any]:
    require(len(report_paths) in {1, 4, 6},
            "aggregate requires exactly one, four, or six case reports")
    cases = [extract_case(path, repo_root) for path in report_paths]
    case_ids = [case["case_id"] for case in cases]
    require(len(set(case_ids)) == len(case_ids), "case report IDs are not unique")
    cases.sort(key=lambda case: case["case_id"])
    case_ids = [case["case_id"] for case in cases]
    if len(cases) == 4:
        require(case_ids == list(PRIMARY_CASE_IDS),
                "four-case aggregate requires exactly e01-e04")
    if len(cases) == 6:
        require(case_ids == list(ALL_CASE_IDS),
                "six-case aggregate requires exactly e01-e06")
        require(four_case_aggregate is not None,
                "six-case aggregate requires a bound four-case aggregate")
    elif four_case_aggregate is not None:
        raise AggregateError(
            "four-case aggregate binding is accepted only for six-case analysis")

    runtime = cases[0]["runtime"]
    require(all(case["runtime"] == runtime for case in cases),
            "case reports do not share one identical subject/runtime")
    subject = str(runtime["subject"])
    exact_semantic = subject == "exact-subject"
    input_bindings = [{
        "case_id": case["case_id"],
        "path": case["display_path"],
        "sha256": case["sha256"],
    } for case in cases]

    directional_threshold = 1 if len(cases) == 1 else 3 if len(cases) == 4 else 4
    families = {
        family: family_analysis(
            cases, family, directional_threshold=directional_threshold)
        for family in FAMILIES
    }
    prior_binding = None
    if len(cases) == 6:
        assert four_case_aggregate is not None
        prior_binding = validate_four_case_aggregate(
            four_case_aggregate, repo_root=repo_root, cases=cases,
            runtime=runtime)
        require(any(
            prior_binding["report"]["families"][family]["classification"] ==
            "AMBIGUOUS4" for family in FAMILIES),
            "six-case reserve was not authorized by four-case ambiguity")
    if len(cases) == 1:
        stage = "ONE_CASE_DESCRIPTIVE"
        status = "INCOMPLETE"
        for row in families.values():
            row.update({
                "classification": "INCOMPLETE",
                "terminal": False,
                "decision_eligible": False,
            })
    elif not exact_semantic:
        stage = "LOCAL_DESCRIPTIVE"
        status = "INCOMPLETE"
        for row in families.values():
            row.update({
                "classification": "INCOMPLETE",
                "terminal": False,
                "decision_eligible": False,
            })
    elif len(cases) == 4:
        stage = "FOUR_CASE_PRIMARY"
        status = "FOUR_CASE_CLASSIFIED"
        for row in families.values():
            classification = classify_four(row)
            row.update({
                "classification": classification,
                "terminal": classification in {"PASS4", "STOP4"},
                "decision_eligible": True,
            })
    else:
        assert prior_binding is not None
        stage = "SIX_CASE_RESERVE"
        status = "SIX_CASE_CLASSIFIED"
        reserve_cases = [case for case in cases
                         if case["case_id"] in RESERVE_CASE_IDS]
        for family, row in families.items():
            prior = prior_binding["report"]["families"][family]["classification"]
            if prior == "AMBIGUOUS4":
                classification = classify_six(row)
                reserve_role = "reclassified_previously_ambiguous_family"
            else:
                classification = prior
                reserve_role = "descriptive_only_prior_terminal_preserved"
            row.update({
                "classification": classification,
                "prior_four_case_classification": prior,
                "reserve_role": reserve_role,
                "terminal": True,
                "decision_eligible": True,
            })
            reserve = family_analysis(
                reserve_cases, family, directional_threshold=2)
            row["reserve_descriptive"] = {
                "case_ids": list(RESERVE_CASE_IDS),
                "per_case": reserve["per_case"],
                "summaries": reserve["summaries"],
                "semantic_reclassification_performed": prior == "AMBIGUOUS4",
            }

    semantic_decisions = exact_semantic and len(cases) in {4, 6}
    result = {
        "schema": AGGREGATE_SCHEMA,
        "design_id": DESIGN_ID,
        "completed_at_utc": utc_now(),
        "stage": stage,
        "status": status,
        "subject": subject,
        "runtime": runtime,
        "case_ids": case_ids,
        "input_reports": input_bindings,
        "semantic_decisions_issued": semantic_decisions,
        "semantic_evidence_eligible": exact_semantic,
        "apparatus_integration_only": not exact_semantic,
        "families": families,
        "paired_family_differences": paired_family_differences(cases),
        "phase_a_utility_damage": {
            case["case_id"]: case["phase_a_fresh_damage"] for case in cases},
        "inference": {
            "p_values_computed": False,
            "confidence_intervals_computed": False,
            "fixed_engineered_cases": True,
            "note": (
                "Reported quantities are descriptive fixed-case summaries and "
                "literal preregistered stopping-rule decisions."),
        },
    }
    if prior_binding is not None:
        result["four_case_aggregate"] = {
            "path": prior_binding["path"],
            "sha256": prior_binding["sha256"],
        }
    return result


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) +
            "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def default_output(repo_root: Path, report: Mapping[str, Any]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    count = len(report["case_ids"])
    return (repo_root / "results/coherent_canary_v12_aggregate" /
            f"coherent-canary-v12-aggregate-n{count}-{report['subject']}-{stamp}.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--four-case-aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = aggregate(
        args.reports, repo_root=args.repo,
        four_case_aggregate=args.four_case_aggregate)
    output = args.output or default_output(args.repo, report)
    print(f"AGGREGATE n={len(report['case_ids'])} subject={report['subject']} "
          f"-> {output}", flush=True)
    write_exclusive_json(output, report)
    print(json.dumps({
        "status": report["status"],
        "families": {family: row["classification"]
                     for family, row in report["families"].items()},
        "output": str(output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
