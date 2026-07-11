#!/usr/bin/env python3
"""Independent Amendment-11 L AND T semantic-release resolver.

This file is intentionally outside the frozen v10 apparatus glob.  It cannot
change a v10 measurement; it can only refuse semantic release.  The exact
launch commit and the release attestation bind its bytes before any semantic
job may start.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


CONTRACT_ID = "COHERENT-STATE-CONJUNCTIVE-AUTHORIZATION-11"
SCHEMA = 1
DESIGN_ID = "coherent-state-gapped-v10"
AMENDMENT_ID = (
    "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9-10")
LADDER_MODEL = "Qwen/Qwen3-0.6B"
LADDER_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
PRODUCTION_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
PRODUCTION_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
LADDER_LAUNCH_COMMIT = "76950df018704764487ae64029635107ce3cdeeb"
V10_APPARATUS_FILE_COUNT = 35
V10_APPARATUS_AGGREGATE_SHA256 = (
    "818a60623c4858f0796865a124d255897325136f147ad611c1810026c9352715")
ELIGIBLE_LADDER_PATH = (
    "results/coherent_state_ladder/"
    "coherent_state_ladder_gapped_v10_Qwen3-0.6B_20260711T123246Z.json")
STAGE_ORDER = (
    "static_provenance",
    "attention_backend",
    "synthetic_schedule_fixtures",
    "committed_case_schedule_fixtures",
    "generated_replay_identity",
    "snapshot_rebuild_identity",
    "physical_causal_mask_identity",
    "future_mutation_identity",
    "position_structure",
    "intervention_propagation",
    "calibration_construction",
    "external_donor_construction",
    "retired_G_delta",
)
FROZEN_ORDER = (
    "c10", "c02", "c01", "c04", "c07", "c11",
    "c05", "c09", "c06", "c12", "c08", "c03",
)
OVERLAY_PATHS = (
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-11.md",
    "COHERENT-STATE-AUTHORIZATION-CLARIFICATION-11A.md",
    "scripts/validate_semantic_release.py",
    "scripts/launch_semantic_release.sh",
    "scripts/job_semantic_release.sh",
)
MAX_ATTESTATION_BYTES = 4_000_000
PREFLIGHT_PATH_RE = re.compile(
    r"^results/v10_release/preflight_[A-Za-z0-9._-]+\.json$")
SAFE_RUN_DIR_RE = re.compile(r"^results/[A-Za-z0-9._/-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ReleaseError(RuntimeError):
    """The conjunctive semantic release is not satisfied."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _payload_sha256(doc: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({
        key: value for key, value in doc.items() if key != "payload_sha256"
    })).hexdigest()


def _verify_payload(doc: dict[str, Any], label: str) -> None:
    if doc.get("payload_sha256") != _payload_sha256(doc):
        raise ReleaseError(f"{label} payload hash differs")


def _seal(doc: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in doc.items()
              if key != "payload_sha256"}
    result["payload_sha256"] = _payload_sha256(result)
    return result


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True,
        text=not binary, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace") if binary else proc.stderr
        raise ReleaseError(
            f"git {' '.join(args)} failed: {stderr.strip()[:300]}")
    return proc.stdout


def _resolve_commit(repo: Path, commit: str, label: str) -> str:
    try:
        return str(_git(repo, "rev-parse", f"{commit}^{{commit}}")).strip()
    except ReleaseError as exc:
        raise ReleaseError(f"{label} commit is not resolvable") from exc


def _require_ancestor(repo: Path, ancestor: str, descendant: str,
                      label: str) -> None:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo, capture_output=True)
    if proc.returncode != 0:
        raise ReleaseError(f"{label} is not on semantic-launch ancestry")


def _commit_bytes(repo: Path, commit: str, relative: str) -> bytes:
    return bytes(_git(repo, "show", f"{commit}:{relative}", binary=True))


def _load_committed_json(repo: Path, commit: str, relative: str) \
        -> tuple[dict[str, Any], bytes]:
    raw = _commit_bytes(repo, commit, relative)
    try:
        doc = json.loads(raw)
    except Exception as exc:
        raise ReleaseError(f"committed JSON is invalid: {relative}") from exc
    if not isinstance(doc, dict):
        raise ReleaseError(f"committed JSON is not an object: {relative}")
    return doc, raw


def _require_identity(doc: dict[str, Any], label: str) -> None:
    if (doc.get("schema") != 2 or doc.get("design_id") != DESIGN_ID or
            doc.get("amendment_id") != AMENDMENT_ID):
        raise ReleaseError(f"{label} scientific identity differs")


def _require_stage_pass(stage: dict[str, Any], name: str) -> None:
    if stage.get("status") != "PASS" or stage.get("passes") is not True:
        raise ReleaseError(f"ladder stage is not terminal PASS: {name}")
    expected = stage.get("expected_coverage")
    observed = stage.get("observed_coverage")
    if (not isinstance(expected, int) or expected < 1 or
            observed != expected):
        raise ReleaseError(f"ladder stage coverage is incomplete: {name}")
    raw = stage.get("raw")
    if not isinstance(raw, dict):
        raise ReleaseError(f"ladder stage raw evidence is absent: {name}")
    if raw.get("failures") not in (None, []):
        raise ReleaseError(f"ladder stage reports failures: {name}")
    if stage.get("failure_evidence") is not None:
        raise ReleaseError(f"ladder stage retains failure evidence: {name}")


def _load_harvest_helpers(repo: Path):
    path = repo / "scripts" / "validate_coherent_harvest.py"
    spec = importlib.util.spec_from_file_location(
        "_amendment11_harvest_helpers", path)
    if spec is None or spec.loader is None:
        raise ReleaseError("cannot load independent harvest helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_local_committed_case_source(
        helper, row: dict[str, Any], cid: str,
        fingerprint: dict[str, Any], repo: Path) -> None:
    """Reuse shared source reconstruction with the local subject revision.

    The tokenizer is loaded and cached under the production revision before
    this wrapper is called. Only the row's exact *model* revision differs from
    the production technical gate.
    """
    prior = helper.MODEL_REVISION
    helper.MODEL_REVISION = LADDER_REVISION
    try:
        helper._validate_committed_case_source(
            row, cid, fingerprint, repo)
    finally:
        helper.MODEL_REVISION = prior


def _validate_ladder_hash_rows(
        value: Any, label: str, *, rows: int | None = None) \
        -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 28:
        raise ReleaseError(f"{label} hash-row coverage differs")
    if [row.get("layer") for row in value] != [str(i) for i in range(28)]:
        raise ReleaseError(f"{label} hash-row order differs")
    for row in value:
        expected = [1, 8, rows, 128] if rows is not None else None
        for prefix in ("k", "v"):
            if row.get(f"{prefix}_dtype") != "torch.bfloat16":
                raise ReleaseError(f"{label} {prefix} dtype differs")
            shape = row.get(f"{prefix}_shape")
            if (not isinstance(shape, list) or len(shape) != 4 or
                    any(not isinstance(item, int) or item < 1 for item in shape) or
                    shape[0] != 1 or shape[1] != 8 or shape[3] != 128 or
                    (expected is not None and shape != expected)):
                raise ReleaseError(f"{label} {prefix} shape differs")
        if row.get("k_shape") != row.get("v_shape"):
            raise ReleaseError(f"{label} K/V shapes differ")
        for key in ("k_sha256", "v_sha256"):
            digest = row.get(key)
            if (not isinstance(digest, str) or len(digest) != 64 or
                    any(char not in "0123456789abcdef" for char in digest.lower())):
                raise ReleaseError(f"{label} {key} differs")
    return value


def _deep_validate_ladder_stages(
        repo: Path, stages: dict[str, dict[str, Any]]) -> None:
    """Recompute local-ladder raw verdicts, never trusting producer booleans."""
    helper = _load_harvest_helpers(repo)
    sys.path.insert(0, str(repo / "src"))
    from coherent_state_calibration import validate_calibration_constructions
    from l_coherent_state_hf import v10_gate_schema
    from validate_coherent_external_donors import validate_with_tokenizer

    schema = v10_gate_schema(expected_attention_layers=28)
    for name in STAGE_ORDER:
        expected = schema[name]
        observed = stages[name]
        for key in ("prerequisites", "threshold", "comparison",
                    "expected_coverage", "metric_names"):
            expected_value = expected.get(key)
            if name == "static_provenance" and key == "metric_names":
                expected_value = ["local_model", "production_tokenizer"]
            if observed.get(key) != expected_value:
                raise ReleaseError(
                    f"ladder stage schema differs: {name}.{key}")

    static = stages["static_provenance"]
    expected_static = {
        "device": "cpu", "dtype": "torch.bfloat16",
        "local_model": LADDER_MODEL,
        "production_tokenizer": PRODUCTION_MODEL,
        "production_tokenizer_revision": PRODUCTION_REVISION,
        "technical_only": True,
    }
    if (static.get("threshold") is not None or
            static.get("comparison") is not None or
            static.get("raw") != expected_static):
        raise ReleaseError("ladder static provenance raw evidence differs")

    attention = stages["attention_backend"]
    fingerprint = (attention.get("raw") or {}).get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ReleaseError("ladder backend fingerprint is absent")
    layers = fingerprint.get("layers")
    if (fingerprint.get("requested_implementation") != "eager" or
            fingerprint.get("expected_layer_count") != 28 or
            not isinstance(layers, list) or len(layers) != 28 or
            [row.get("layer_index") for row in layers] != list(range(28)) or
            any(row.get("resolved_implementation") != "eager" for row in layers)):
        raise ReleaseError("ladder backend layer evidence differs")
    unhashed = {key: value for key, value in fingerprint.items()
                if key != "sha256"}
    if fingerprint.get("sha256") != hashlib.sha256(
            _canonical(unhashed)).hexdigest():
        raise ReleaseError("ladder backend fingerprint hash differs")

    synthetic = stages["synthetic_schedule_fixtures"]
    if (helper._finite(synthetic.get("threshold"), "synthetic threshold") !=
            5e-4 or synthetic.get("comparison") != "<="):
        raise ReleaseError("ladder synthetic threshold differs")
    raw = synthetic.get("raw") or {}
    provenance = raw.get("fixture_provenance") or {}
    if (provenance.get("literal") != helper.FROZEN_FIXTURE_LITERAL or
            provenance.get("pool_token_ids") != helper.FROZEN_FIXTURE_POOL or
            provenance.get("pool_sha256") != helper._sha256_ints(
                helper.FROZEN_FIXTURE_POOL, "synthetic.pool") or
            provenance.get("margin_token_ids") != helper.FROZEN_MARGIN_IDS or
            provenance.get("continuation_token_id") !=
            helper.FROZEN_CONTINUATION_ID or
            provenance.get("pool_contains_special_token") is not False):
        raise ReleaseError("ladder synthetic fixture provenance differs")
    rows = raw.get("contiguous")
    if not isinstance(rows, list) or len(rows) != len(helper.SYNTHETIC_LENGTHS):
        raise ReleaseError("ladder synthetic fixture coverage differs")
    aggregates = []
    try:
        for index, (row, length, partitions) in enumerate(zip(
                rows, helper.SYNTHETIC_LENGTHS,
                helper.SYNTHETIC_PARTITIONS)):
            if (row.get("length") != length or
                    row.get("reference_partition") != list(partitions[0]) or
                    row.get("alternative_partition") != list(partitions[1])):
                raise ValueError(f"synthetic fixture {index} declaration differs")
            token_ids = [helper.FROZEN_FIXTURE_POOL[i % len(
                helper.FROZEN_FIXTURE_POOL)] for i in range(length)]
            if row.get("token_ids_sha256") != helper._sha256_ints(
                    token_ids, f"synthetic[{length}].tokens"):
                raise ValueError("synthetic token hash differs")
            aggregates.append(helper._validate_schedule_measurement(
                row, layers=28, tolerance=5e-4,
                label=f"ladder.synthetic[{length}]"))
        gap = raw.get("logical_gap")
        if not isinstance(gap, dict):
            raise ValueError("logical gap absent")
        positions = list(range(32)) + list(range(8192, 8224))
        tokens = [helper.FROZEN_FIXTURE_POOL[i % len(
            helper.FROZEN_FIXTURE_POOL)] for i in range(64)]
        if (gap.get("logical_positions") != positions or
                gap.get("physical_cache_positions") != list(range(64)) or
                gap.get("reference_partition") != [32, 32] or
                gap.get("alternative_partition") != [32] + [1] * 32 or
                gap.get("continuation_logical_position") != 8224 or
                gap.get("logical_as_cache_position_rejected") is not True or
                gap.get("full_attention_over_physically_prior_rows") is not True or
                gap.get("token_ids_sha256") != helper._sha256_ints(
                    tokens, "logical_gap.tokens") or
                gap.get("logical_positions_sha256") != helper._sha256_ints(
                    positions, "logical_gap.positions")):
            raise ValueError("logical gap evidence differs")
        aggregates.append(helper._validate_schedule_measurement(
            gap, layers=28, tolerance=5e-4, label="ladder.logical_gap"))
    except ValueError as exc:
        raise ReleaseError(str(exc)) from exc
    if (synthetic.get("observed_aggregate") != max(aggregates) or
            raw.get("passes") is not True or raw.get("failures") not in ([], None)):
        raise ReleaseError("ladder synthetic aggregate/verdict differs")

    committed = stages["committed_case_schedule_fixtures"]
    if (helper._finite(committed.get("threshold"), "case threshold") != 5e-4 or
            committed.get("comparison") != "<="):
        raise ReleaseError("ladder committed-case threshold differs")
    case_raw = committed.get("raw") or {}
    case_rows = case_raw.get("rows")
    if not isinstance(case_rows, list) or len(case_rows) != 12:
        raise ReleaseError("ladder committed-case rows are absent")
    tokenizer = helper._validation_tokenizer()
    inventory_rows = []
    for cid in FROZEN_ORDER:
        path = repo / "data" / "synthetic" / f"{cid}.json"
        blob = path.read_bytes()
        inventory_rows.append({
            "path": f"data/synthetic/{cid}.json", "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest()})
    source_fingerprint = {
        "input_inventory": {"files": inventory_rows},
        "subject_metadata": {
            "tokenizer_vocab_sha256": helper._canonical_json_sha256(
                tokenizer.get_vocab()),
            "chat_template_sha256": hashlib.sha256(
                str(tokenizer.chat_template).encode()).hexdigest(),
        },
        "summary_request_sha256": hashlib.sha256(
            helper.SUMMARY_REQUEST.encode()).hexdigest(),
    }
    case_aggregates = []
    try:
        for index, (row, cid) in enumerate(zip(case_rows, FROZEN_ORDER), 1):
            if (row.get("conversation_id") != cid or
                    row.get("order_position") != index):
                raise ValueError(f"committed-case identity differs: {cid}")
            _validate_local_committed_case_source(
                helper,
                row, cid, source_fingerprint, repo)
            case_aggregates.append(helper._validate_schedule_measurement(
                row, layers=28, tolerance=5e-4, label=f"ladder.case[{cid}]"))
    except ValueError as exc:
        raise ReleaseError(str(exc)) from exc
    if (case_raw.get("frozen_order") != list(FROZEN_ORDER) or
            case_raw.get("coverage_exact") is not True or
            committed.get("observed_aggregate") != max(case_aggregates)):
        raise ReleaseError("ladder committed-case aggregate/order differs")

    generated = stages["generated_replay_identity"]
    gen_raw = generated.get("raw") or {}
    gen_per = gen_raw.get("per_layer")
    if (not isinstance(gen_per, list) or len(gen_per) != 28 or
            [row.get("layer") for row in gen_per] != list(range(28))):
        raise ReleaseError("ladder generated/replay layer coverage differs")
    gen_k = max(helper._finite(row.get("k_max_abs"), "generated K")
                for row in gen_per)
    gen_v = max(helper._finite(row.get("v_max_abs"), "generated V")
                for row in gen_per)
    gen_values = [
        helper._finite(gen_raw.get("token_logprob_max_abs"), "generated lp"),
        gen_k, gen_v]
    if (gen_raw.get("k_max_abs") != gen_k or gen_raw.get("v_max_abs") != gen_v or
            generated.get("observed_aggregate") != max(gen_values) or
            max(gen_values) > 1e-4 or gen_raw.get("passes") is not True or
            gen_raw.get("tolerance") != 1e-4 or
            gen_raw.get("comparison") != "<=" or gen_k != 0.0 or gen_v != 0.0 or
            any(row.get("k_max_abs") != 0.0 or row.get("v_max_abs") != 0.0
                for row in gen_per)):
        raise ReleaseError("ladder generated/replay aggregate differs")

    for name, fields in (
        ("snapshot_rebuild_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("physical_causal_mask_identity", ("logits_max_abs", "k_max_abs", "v_max_abs")),
        ("future_mutation_identity", ("earlier_logits_max_abs", "earlier_cache_max_abs")),
    ):
        stage = stages[name]
        stage_raw = stage.get("raw") or {}
        values = [helper._finite(stage_raw.get(field), f"{name}.{field}")
                  for field in fields]
        per_layer = stage_raw.get("per_layer")
        if (not isinstance(per_layer, list) or len(per_layer) != 28 or
                [row.get("layer") for row in per_layer] != list(range(28))):
            raise ReleaseError(f"ladder {name} layer coverage differs")
        per_k = max(helper._finite(row.get("k_max_abs"), f"{name}.K")
                    for row in per_layer)
        per_v = max(helper._finite(row.get("v_max_abs"), f"{name}.V")
                    for row in per_layer)
        if name == "future_mutation_identity":
            consistent = values[1] == max(per_k, per_v)
        else:
            consistent = values[1] == per_k and values[2] == per_v
        if (not consistent or stage.get("observed_aggregate") != max(values) or
                max(values) > 1e-4):
            raise ReleaseError(f"ladder {name} aggregate differs")

    position = stages["position_structure"].get("raw") or {}
    for key in ("common_summary_start", "wrong_prefix_length_equal",
                "post_summary_nonempty"):
        if position.get(key) is not True:
            raise ReleaseError(f"ladder position evidence lacks {key}")
    for key in ("altered_structure_failure_injection",
                "wrong_position_failure_injection"):
        if (position.get(key) or {}).get("rejected") is not True:
            raise ReleaseError(f"ladder position injection failed: {key}")
    physical = position.get("physical_cache_positions")
    context = position.get("context_position_ids")
    prefix = position.get("prefix_position_ids")
    summary = position.get("summary_position_ids")
    post = position.get("post_summary_position_ids")
    source_start = position.get("source_summary_start")
    physical_start = position.get("physical_summary_start")
    physical_end = position.get("physical_summary_end")
    system_end = position.get("system_end")
    request_start = position.get("request_logical_start")
    if (not all(isinstance(value, int) for value in (
            source_start, physical_start, physical_end, system_end, request_start)) or
            prefix != list(range(system_end)) + list(range(request_start, source_start)) or
            not isinstance(summary, list) or not summary or
            summary != list(range(source_start, source_start + len(summary))) or
            physical_start != len(prefix) or physical_end != physical_start + len(summary) or
            not isinstance(post, list) or post != list(range(
                source_start + len(summary), source_start + len(summary) + len(post))) or
            context != prefix + summary + post or
            position.get("logical_next_position") != context[-1] + 1 or
            position.get("logical_gap") != source_start - physical_start or
            physical != list(range(len(context)))):
        raise ReleaseError("ladder position schedule evidence differs")
    correct = position.get("correct_prefix_ids")
    wrong = position.get("wrong_prefix_ids")
    structural = position.get("wrong_structural_position_ids")
    content = position.get("wrong_content_position_ids")
    if (not isinstance(correct, list) or not isinstance(wrong, list) or
            len(correct) != len(wrong) or not isinstance(structural, list) or
            not isinstance(content, list) or set(structural) & set(content) or
            sorted(structural + content) != list(range(len(correct))) or
            any(correct[i] != wrong[i] for i in structural) or
            not any(correct[i] != wrong[i] for i in content) or
            position.get("correct_prefix_sha256") != helper._sha256_ints(
                correct, "position.correct") or
            position.get("wrong_prefix_sha256") != helper._sha256_ints(
                wrong, "position.wrong") or
            position.get("wrong_structural_positions") != len(structural) or
            position.get("wrong_content_positions") != len(content) or
            position.get("wrong_structural_positions_sha256") !=
            helper._sha256_ints(structural, "position.structural") or
            position.get("wrong_content_positions_sha256") !=
            helper._sha256_ints(content, "position.content")):
        raise ReleaseError("ladder position wrong-source evidence differs")
    for values_key, hash_key in (
            ("context_ids", "context_ids_sha256"),
            ("summary_ids", "summary_ids_sha256")):
        if position.get(hash_key) != helper._sha256_ints(
                position.get(values_key), f"position.{values_key}"):
            raise ReleaseError(f"ladder position hash differs: {hash_key}")

    intervention = stages["intervention_propagation"].get("raw") or {}
    for key in ("fresh_self_replacement",
                "correct_insert_and_non_summary_preservation",
                "wrong_insert_and_non_summary_preservation",
                "per_arm_fork_at_summary_boundary",
                "independently_recomputed_identical_tail_lengths"):
        if intervention.get(key) is not True:
            raise ReleaseError(f"ladder intervention evidence lacks {key}")
    if (intervention.get("pre_tailed_failure_injection") or {}).get(
            "rejected") is not True:
        raise ReleaseError("ladder intervention failure injection did not reject")
    attempts = intervention.get("sensitivity_attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ReleaseError("ladder intervention sensitivity attempts are absent")
    sensitivity = tail_changed = False
    for index, attempt in enumerate(attempts):
        epsilon = helper._finite(attempt.get("epsilon"), f"epsilon[{index}]")
        logits = helper._finite(
            attempt.get("fixed_continuation_logits_max_abs"), "logits change")
        tail = helper._finite(
            attempt.get("recomputed_post_summary_kv_max_abs"), "tail change")
        if epsilon not in (0.1, 0.3, 1.0, 3.0) or logits < 0 or tail < 0:
            raise ReleaseError("ladder intervention sensitivity values differ")
        sensitivity |= logits > 1e-4
        tail_changed |= tail > 0
    if (intervention.get("downstream_sensitivity") is not sensitivity or
            intervention.get("recomputed_tail_changed") is not tail_changed or
            not sensitivity or not tail_changed):
        raise ReleaseError("ladder intervention sensitivity recomputation differs")
    span = intervention.get("summary_span") or {}
    start, end = span.get("start"), span.get("end")
    boundary_lengths = intervention.get("boundary_lengths") or {}
    full_lengths = intervention.get("full_lengths") or {}
    if (not isinstance(start, int) or not isinstance(end, int) or
            not 0 <= start < end or
            set(boundary_lengths) != {"fresh", "self", "correct", "wrong"} or
            len(set(boundary_lengths.values())) != 1 or
            next(iter(boundary_lengths.values())) != end or
            set(full_lengths) != {"fresh", "correct", "wrong"} or
            len(set(full_lengths.values())) != 1 or
            next(iter(full_lengths.values())) <= end):
        raise ReleaseError("ladder intervention span/length evidence differs")
    hashes = intervention.get("hashes") or {}
    required_hashes = {
        "fresh_boundary", "self_boundary", "correct_boundary", "wrong_boundary",
        "fresh_before_summary", "correct_before_summary", "wrong_before_summary",
        "correct_source_summary", "wrong_source_summary",
        "correct_inserted_summary", "wrong_inserted_summary", "full_fresh",
        "full_correct", "full_wrong", "fresh_post_summary",
        "correct_post_summary", "wrong_post_summary"}
    if set(hashes) != required_hashes:
        raise ReleaseError("ladder intervention hash field set differs")
    hash_widths = {
        "fresh_boundary": end, "self_boundary": end,
        "correct_boundary": end, "wrong_boundary": end,
        "fresh_before_summary": start, "correct_before_summary": start,
        "wrong_before_summary": start,
        "correct_source_summary": end - start,
        "wrong_source_summary": end - start,
        "correct_inserted_summary": end - start,
        "wrong_inserted_summary": end - start,
        "full_fresh": full_lengths["fresh"],
        "full_correct": full_lengths["correct"],
        "full_wrong": full_lengths["wrong"],
        "fresh_post_summary": full_lengths["fresh"] - end,
        "correct_post_summary": full_lengths["correct"] - end,
        "wrong_post_summary": full_lengths["wrong"] - end,
    }
    for key in required_hashes:
        _validate_ladder_hash_rows(
            hashes[key], f"intervention.{key}", rows=hash_widths[key])
    if (hashes["self_boundary"] != hashes["fresh_boundary"] or
            hashes["correct_before_summary"] != hashes["fresh_before_summary"] or
            hashes["wrong_before_summary"] != hashes["fresh_before_summary"] or
            hashes["correct_inserted_summary"] != hashes["correct_source_summary"] or
            hashes["wrong_inserted_summary"] != hashes["wrong_source_summary"] or
            (hashes["correct_post_summary"] == hashes["fresh_post_summary"] and
             hashes["wrong_post_summary"] == hashes["fresh_post_summary"])):
        raise ReleaseError("ladder intervention hash equalities differ")

    expected_calibration = validate_calibration_constructions(tokenizer)
    if stages["calibration_construction"].get("raw") != expected_calibration:
        raise ReleaseError("ladder calibration construction differs from recomputation")
    expected_donors = validate_with_tokenizer(
        tokenizer, repo / "data" / "synthetic",
        resolved_revision=PRODUCTION_REVISION)
    if stages["external_donor_construction"].get("raw") != expected_donors:
        raise ReleaseError("ladder external donors differ from recomputation")
    if stages["retired_G_delta"].get("raw") != {
            "retired_by": "Amendment 4", "executed": False,
            "authorizes_run": False}:
        raise ReleaseError("ladder retired-arm evidence differs")


def verify_apparatus_bridge(repo: Path, current_apparatus: dict[str, Any]) \
        -> dict[str, Any]:
    launch = _resolve_commit(repo, LADDER_LAUNCH_COMMIT, "ladder launch")
    files = current_apparatus.get("files")
    if (not isinstance(files, list) or
            current_apparatus.get("file_count") != V10_APPARATUS_FILE_COUNT or
            len(files) != V10_APPARATUS_FILE_COUNT or
            current_apparatus.get("aggregate_sha256") !=
            V10_APPARATUS_AGGREGATE_SHA256):
        raise ReleaseError("current v10 apparatus inventory is malformed")
    for row in files:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str):
            raise ReleaseError("current apparatus contains an invalid path")
        raw = _commit_bytes(repo, launch, relative)
        if (len(raw) != row.get("bytes") or
                hashlib.sha256(raw).hexdigest() != row.get("sha256")):
            raise ReleaseError(
                f"v10 apparatus differs from ladder launch: {relative}")
    return {
        "ladder_launch_commit": launch,
        "apparatus_file_count": current_apparatus.get("file_count"),
        "apparatus_aggregate_sha256":
            current_apparatus.get("aggregate_sha256"),
        "all_inventoried_bytes_exact": True,
    }


def validate_ladder_commit(
    repo: Path,
    *,
    result_commit: str,
    ladder_path: str,
    semantic_launch_commit: str,
    current_apparatus: dict[str, Any],
) -> dict[str, Any]:
    if ladder_path != ELIGIBLE_LADDER_PATH:
        raise ReleaseError("ladder path is not the Amendment-11 eligible attempt")
    result = _resolve_commit(repo, result_commit, "ladder result")
    launch = _resolve_commit(repo, semantic_launch_commit, "semantic launch")
    _require_ancestor(repo, result, launch, "ladder result")
    bridge = verify_apparatus_bridge(repo, current_apparatus)

    manifest, manifest_raw = _load_committed_json(
        repo, result, ladder_path)
    _verify_payload(manifest, "ladder manifest")
    _require_identity(manifest, "ladder manifest")
    if (manifest.get("status") != "PASS" or
            manifest.get("model") != LADDER_MODEL or
            manifest.get("resolved_revision") != LADDER_REVISION or
            manifest.get("dtype") != "torch.bfloat16" or
            manifest.get("device") != "cpu"):
        raise ReleaseError("ladder terminal subject or status differs")

    gate = manifest.get("loaded_gapped_production_gate")
    if not isinstance(gate, dict):
        gate = (manifest.get("diagnostics") or {}).get(
            "loaded_gapped_production_gate")
    if not isinstance(gate, dict):
        raise ReleaseError("ladder terminal gate is absent")
    _require_identity(gate, "ladder gate")
    if (gate.get("externalized") is not True or
            gate.get("status") != "PASS" or gate.get("passes") is not True or
            gate.get("technical_only") is not True or
            gate.get("stage_order") != list(STAGE_ORDER) or
            gate.get("failures") != [] or gate.get("failure") is not None):
        raise ReleaseError("ladder terminal gate envelope differs")
    refs = gate.get("stage_refs")
    if not isinstance(refs, dict) or set(refs) != set(STAGE_ORDER):
        raise ReleaseError("ladder stage references are not exact")

    artifact_rows = manifest.get("artifact_files")
    if (not isinstance(artifact_rows, list) or
            len(artifact_rows) != len(STAGE_ORDER) or
            any(not isinstance(row, dict) for row in artifact_rows)):
        raise ReleaseError("ladder artifact file inventory is absent")
    artifact_paths = [row.get("path") for row in artifact_rows]
    if (any(not isinstance(path, str) for path in artifact_paths) or
            len(set(artifact_paths)) != len(STAGE_ORDER)):
        raise ReleaseError("ladder artifact file inventory is not unique")
    artifact_by_path = {
        row["path"]: row for row in artifact_rows
    }
    if set(artifact_by_path) != {
            ref.get("path") for ref in refs.values() if isinstance(ref, dict)}:
        raise ReleaseError("ladder artifact/stage path sets differ")

    stages: dict[str, dict[str, Any]] = {}
    parent = Path(ladder_path).parent
    for name in STAGE_ORDER:
        ref = refs[name]
        relative_name = ref.get("path") if isinstance(ref, dict) else None
        if (not isinstance(relative_name, str) or
                Path(relative_name).name != relative_name):
            raise ReleaseError(f"ladder stage path is invalid: {name}")
        relative = (parent / relative_name).as_posix()
        sidecar, raw = _load_committed_json(repo, result, relative)
        _verify_payload(sidecar, f"ladder stage {name}")
        _require_identity(sidecar, f"ladder stage {name}")
        if (sidecar.get("artifact_kind") != "ladder_gate_stage" or
                sidecar.get("stage_name") != name):
            raise ReleaseError(f"ladder stage identity differs: {name}")
        if (ref.get("byte_count") != len(raw) or
                ref.get("raw_file_sha256") != hashlib.sha256(raw).hexdigest() or
                ref.get("payload_sha256") != sidecar.get("payload_sha256")):
            raise ReleaseError(f"ladder stage reference differs: {name}")
        artifact = artifact_by_path[relative_name]
        if (artifact.get("byte_count") != len(raw) or
                artifact.get("raw_file_sha256") !=
                hashlib.sha256(raw).hexdigest()):
            raise ReleaseError(f"ladder artifact inventory differs: {name}")
        stage = sidecar.get("stage")
        if not isinstance(stage, dict):
            raise ReleaseError(f"ladder stage payload is absent: {name}")
        _require_stage_pass(stage, name)
        stages[name] = stage

    _deep_validate_ladder_stages(repo, stages)

    static = stages["static_provenance"]
    if (static.get("expected_coverage"), static.get("observed_coverage")) != (1, 1):
        raise ReleaseError("ladder static-provenance coverage differs")
    expected_static = {
        "device": "cpu", "dtype": "torch.bfloat16",
        "local_model": LADDER_MODEL,
        "production_tokenizer": PRODUCTION_MODEL,
        "production_tokenizer_revision": PRODUCTION_REVISION,
        "technical_only": True,
    }
    if static.get("raw") != expected_static:
        raise ReleaseError("ladder static provenance differs")

    attention = stages["attention_backend"]
    if (attention.get("expected_coverage"),
            attention.get("observed_coverage")) != (28, 28):
        raise ReleaseError("ladder eager-attention coverage differs")
    fingerprint = (attention.get("raw") or {}).get("fingerprint")
    layers = fingerprint.get("layers") if isinstance(fingerprint, dict) else None
    if (fingerprint.get("requested_implementation") != "eager" or
            fingerprint.get("expected_layer_count") != 28 or
            not isinstance(layers, list) or len(layers) != 28 or
            any(row.get("resolved_implementation") != "eager"
                for row in layers if isinstance(row, dict)) or
            any(not isinstance(row, dict) for row in layers)):
        raise ReleaseError("ladder eager-attention fingerprint differs")

    synthetic = stages["synthetic_schedule_fixtures"]
    if (synthetic.get("expected_coverage"),
            synthetic.get("observed_coverage")) != (7, 7):
        raise ReleaseError("ladder synthetic coverage differs")
    committed = stages["committed_case_schedule_fixtures"]
    if (committed.get("expected_coverage"),
            committed.get("observed_coverage")) != (12, 12):
        raise ReleaseError("ladder committed-case coverage differs")
    committed_raw = committed.get("raw") or {}
    rows = committed_raw.get("rows")
    if (committed_raw.get("frozen_order") != list(FROZEN_ORDER) or
            not isinstance(rows, list) or len(rows) != 12 or
            tuple(row.get("conversation_id") for row in rows) != FROZEN_ORDER or
            any(row.get("status") != "PASS" or row.get("passes") is not True
                for row in rows if isinstance(row, dict)) or
            any(not isinstance(row, dict) for row in rows)):
        raise ReleaseError("ladder committed-case rows differ")

    return {
        "status": "PASS",
        "result_commit": result,
        "path": ladder_path,
        "manifest_raw_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_payload_sha256": manifest["payload_sha256"],
        "stage_payload_sha256": {
            name: refs[name]["payload_sha256"] for name in STAGE_ORDER},
        "committed_cases": 12,
        "synthetic_fixtures": 7,
        "attention_layers": 28,
        "bridge": bridge,
    }


def _overlay_inventory(repo: Path, commit: str) -> dict[str, str]:
    rows = {}
    for relative in OVERLAY_PATHS:
        raw = _commit_bytes(repo, commit, relative)
        rows[relative] = hashlib.sha256(raw).hexdigest()
    return rows


def build_release_attestation(
    repo: Path,
    *,
    evidence_commit: str,
    verification_launch_commit: str | None = None,
    ladder_result_commit: str,
    ladder_path: str,
    technical_result_commit: str,
    technical_run_dir: str,
) -> dict[str, Any]:
    evidence = _resolve_commit(repo, evidence_commit, "release evidence")
    verification_launch = _resolve_commit(
        repo, verification_launch_commit or evidence,
        "release verification launch")
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != verification_launch:
        raise ReleaseError(
            f"HEAD {head} != release verification launch {verification_launch}")
    _require_ancestor(repo, evidence, verification_launch, "release evidence")

    sys.path.insert(0, str(repo / "src"))
    from coherent_state_integrity import (  # pylint: disable=import-outside-toplevel
        apparatus_inventory, verify_prior_technical_authorization)

    apparatus = apparatus_inventory(repo)
    ladder = validate_ladder_commit(
        repo, result_commit=ladder_result_commit, ladder_path=ladder_path,
        semantic_launch_commit=verification_launch,
        current_apparatus=apparatus)
    technical = verify_prior_technical_authorization(
        repo, repo / technical_run_dir, technical_result_commit,
        verification_launch, apparatus)
    return _seal({
        "schema": SCHEMA,
        "authorization_contract_id": CONTRACT_ID,
        "status": "PASS",
        "authorization_expression": "L AND T",
        "authorization_state": "SEMANTIC_RELEASE_ELIGIBLE",
        "evidence_commit": evidence,
        "ladder": ladder,
        "technical": {
            "status": "PASS",
            "result_commit": technical.result_commit,
            "run_dir": technical.run_dir,
            "gate_payload_sha256": technical.gate_payload_sha256,
            "raw_sha256": technical.raw_sha256,
            "harvest_payload_sha256":
                technical.harvest["attestation"]["payload_sha256"],
        },
        "apparatus_aggregate_sha256": apparatus["aggregate_sha256"],
        "overlay_sha256": _overlay_inventory(repo, evidence),
        "semantic_outcomes_observed": 0,
    })


def _atomic_write(path: Path, doc: dict[str, Any]) -> None:
    if path.exists():
        raise ReleaseError(f"refusing to overwrite release attestation: {path}")
    raw = _canonical(doc) + b"\n"
    if len(raw) >= MAX_ATTESTATION_BYTES:
        raise ReleaseError(
            f"release attestation is not commit-safe: {len(raw)} bytes")
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def _require_safe_technical_binding(commit: str, run_dir: str) -> None:
    if not COMMIT_RE.fullmatch(commit):
        raise ReleaseError("attested technical result commit is not exact 40-hex")
    if (not SAFE_RUN_DIR_RE.fullmatch(run_dir) or ".." in Path(run_dir).parts or
            "//" in run_dir):
        raise ReleaseError("attested technical run directory is not shell-safe")


def verify_committed_attestation(
    repo: Path, attestation_path: str, expected_launch_commit: str) \
        -> dict[str, Any]:
    launch = _resolve_commit(repo, expected_launch_commit, "expected launch")
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != launch:
        raise ReleaseError(f"HEAD {head} != expected launch {launch}")
    origin = str(_git(repo, "rev-parse", "origin/trunk")).strip()
    if origin != launch:
        raise ReleaseError(f"origin/trunk {origin} != expected launch {launch}")
    if str(_git(repo, "status", "--porcelain")).strip():
        raise ReleaseError("semantic launch worktree is not clean")
    attestation, raw = _load_committed_json(
        repo, launch, attestation_path)
    _verify_payload(attestation, "release attestation")
    if (attestation.get("schema") != SCHEMA or
            attestation.get("authorization_contract_id") != CONTRACT_ID or
            attestation.get("status") != "PASS" or
            attestation.get("authorization_expression") != "L AND T" or
            attestation.get("semantic_outcomes_observed") != 0):
        raise ReleaseError("release attestation identity or state differs")
    evidence = _resolve_commit(
        repo, attestation.get("evidence_commit", ""), "attested evidence")
    _require_ancestor(repo, evidence, launch, "attested evidence")
    recomputed = build_release_attestation(
        repo, evidence_commit=evidence,
        verification_launch_commit=launch,
        ladder_result_commit=attestation["ladder"]["result_commit"],
        ladder_path=attestation["ladder"]["path"],
        technical_result_commit=attestation["technical"]["result_commit"],
        technical_run_dir=attestation["technical"]["run_dir"],
    )
    if recomputed != attestation:
        raise ReleaseError("committed release attestation differs from recomputation")
    current_overlay = {
        relative: hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        for relative in OVERLAY_PATHS}
    if current_overlay != attestation.get("overlay_sha256"):
        raise ReleaseError("authorization overlay changed since evidence commit")
    technical_commit = attestation["technical"]["result_commit"]
    technical_run_dir = attestation["technical"]["run_dir"]
    _require_safe_technical_binding(technical_commit, technical_run_dir)
    return {
        "status": "PASS",
        "authorization_contract_id": CONTRACT_ID,
        "launch_commit": launch,
        "attestation_path": attestation_path,
        "attestation_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "attestation_payload_sha256": attestation["payload_sha256"],
        "authorization_expression": "L AND T",
        "technical_result_commit": technical_commit,
        "technical_run_dir": technical_run_dir,
    }


def _verify_committed_result_tree(
    repo: Path, run_dir: str, result_commit: str, release_head: str,
) -> tuple[str, Path, list[dict[str, Any]]]:
    result = _resolve_commit(repo, result_commit, "semantic result")
    head = _resolve_commit(repo, release_head, "release head")
    _require_ancestor(repo, result, head, "semantic result")
    root = (repo / run_dir).resolve()
    try:
        relative_root = root.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ReleaseError("semantic result directory is outside repository") from exc
    tree = str(_git(
        repo, "ls-tree", "-r", "--name-only", result, "--", relative_root))
    tree_paths = sorted(row for row in tree.splitlines() if row)
    disk_paths = sorted(
        path.relative_to(repo).as_posix()
        for path in root.rglob("*") if path.is_file())
    if not tree_paths or tree_paths != disk_paths:
        raise ReleaseError("committed semantic result tree differs from disk")
    rows = []
    for relative in tree_paths:
        committed = _commit_bytes(repo, result, relative)
        disk = (repo / relative).read_bytes()
        if committed != disk:
            raise ReleaseError(f"semantic result bytes differ: {relative}")
        rows.append({
            "path": relative, "bytes": len(disk),
            "sha256": hashlib.sha256(disk).hexdigest()})
    return result, root, rows


def _require_semantic_technical_binding(
        manifest: dict[str, Any], preflight_doc: dict[str, Any]) -> None:
    prior = manifest.get("semantic_authorization") or {}
    prior_harvest = prior.get("harvest") or {}
    expected = preflight_doc.get("technical") or {}
    if (prior.get("result_commit") != expected.get("result_commit") or
            prior.get("run_dir") != expected.get("run_dir") or
            prior.get("gate_payload_sha256") !=
            expected.get("gate_payload_sha256") or
            prior.get("raw_sha256") != expected.get("raw_sha256") or
            prior_harvest.get("payload_sha256") !=
            expected.get("harvest_payload_sha256")):
        raise ReleaseError(
            "semantic result used a different technical authorization than preflight")


def validate_semantic_result(
    repo: Path, *, run_dir: str, result_commit: str, release_head: str,
    preflight_path: str,
) -> dict[str, Any]:
    result, root, rows = _verify_committed_result_tree(
        repo, run_dir, result_commit, release_head)
    sibling = Path(f"{root}.harvest_validation.json")
    try:
        sibling_relative = sibling.relative_to(repo).as_posix()
    except ValueError as exc:
        raise ReleaseError("semantic harvest attestation is outside repository") from exc
    committed_sibling = _commit_bytes(repo, result, sibling_relative)
    if not sibling.is_file() or sibling.read_bytes() != committed_sibling:
        raise ReleaseError("semantic harvest attestation is not exact committed bytes")
    sibling_doc = json.loads(committed_sibling)
    _verify_payload(sibling_doc, "semantic harvest attestation")
    if (sibling_doc.get("status") != "PASS" or
            sibling_doc.get("mode") != "complete" or
            sibling_doc.get("run_dir") != run_dir):
        raise ReleaseError("semantic harvest attestation is not COMPLETE PASS")

    validator = repo / "scripts" / "validate_coherent_harvest.py"
    proc = subprocess.run(
        [sys.executable, str(validator), str(root), "complete", "--read-only"],
        cwd=repo, text=True, capture_output=True)
    if proc.returncode != 0:
        raise ReleaseError(
            f"independent semantic harvest failed: {proc.stderr.strip()[:500]}")
    try:
        rerun = json.loads(proc.stdout)
    except Exception as exc:
        raise ReleaseError("semantic harvest validator emitted invalid JSON") from exc
    if rerun.get("status") != "PASS" or rerun.get("mode") != "complete":
        raise ReleaseError("independent semantic harvest is not COMPLETE PASS")

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseError("semantic manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    _verify_payload(manifest, "semantic manifest")
    _require_identity(manifest, "semantic manifest")
    static = manifest.get("fingerprint_static") or manifest.get("fingerprint") or {}
    semantic_launch = static.get("code_commit")
    if not isinstance(semantic_launch, str):
        raise ReleaseError("semantic manifest lacks launch commit")
    semantic_launch = _resolve_commit(repo, semantic_launch, "semantic launch")
    _require_ancestor(repo, semantic_launch, result, "semantic launch")

    current_preflight = (repo / preflight_path).read_bytes()
    launch_preflight = _commit_bytes(repo, semantic_launch, preflight_path)
    if current_preflight != launch_preflight:
        raise ReleaseError(
            "release preflight was absent or different at semantic launch")
    preflight_doc = json.loads(current_preflight)
    if _overlay_inventory(repo, semantic_launch) != \
            preflight_doc.get("overlay_sha256"):
        raise ReleaseError("authorization overlay differed at semantic launch")
    expected_technical = preflight_doc.get("technical") or {}
    _require_semantic_technical_binding(manifest, preflight_doc)
    expected_receipt = (
        "SEMANTIC_RELEASE_OUTER_PASS "
        f"launch={semantic_launch} packet={preflight_path} "
        f"raw={hashlib.sha256(current_preflight).hexdigest()} "
        f"payload={preflight_doc.get('payload_sha256')} "
        f"technical_commit={expected_technical.get('result_commit')} "
        f"technical_run_dir={expected_technical.get('run_dir')}")
    log_lines = (root / "job.log").read_text(errors="replace").splitlines()
    receipts = [line for line in log_lines
                if line.startswith("SEMANTIC_RELEASE_OUTER_PASS ")]
    if receipts != [expected_receipt]:
        raise ReleaseError("semantic result lacks exact outer-release receipt")
    return {
        "status": "PASS",
        "result_commit": result,
        "run_dir": run_dir,
        "semantic_launch_commit": semantic_launch,
        "tree_files": rows,
        "harvest_path": sibling_relative,
        "harvest_raw_sha256": hashlib.sha256(committed_sibling).hexdigest(),
        "harvest_payload_sha256": sibling_doc["payload_sha256"],
        "independent_harvest": rerun,
        "preflight_at_launch_raw_sha256":
            hashlib.sha256(launch_preflight).hexdigest(),
        "outer_release_receipt": expected_receipt,
    }


def build_final_release_attestation(
    repo: Path, *, preflight_path: str, semantic_run_dir: str,
    semantic_result_commit: str, release_head: str,
) -> dict[str, Any]:
    head = _resolve_commit(repo, release_head, "final release head")
    current_head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if current_head != head:
        raise ReleaseError(f"HEAD {current_head} != final release head {head}")
    preflight_verification = verify_committed_attestation(
        repo, preflight_path, head)
    preflight_doc = json.loads((repo / preflight_path).read_text())
    semantic = validate_semantic_result(
        repo, run_dir=semantic_run_dir,
        result_commit=semantic_result_commit, release_head=head,
        preflight_path=preflight_path)
    return _seal({
        "schema": SCHEMA,
        "authorization_contract_id": CONTRACT_ID,
        "status": "PASS",
        "release_state": "SEMANTIC_RESULT_RELEASED",
        "authorization_expression":
            "L AND T AND SEMANTIC_HARVEST_PASS",
        "release_head": head,
        "preflight": {
            "path": preflight_path,
            "raw_sha256": hashlib.sha256(
                (repo / preflight_path).read_bytes()).hexdigest(),
            "payload_sha256": preflight_doc["payload_sha256"],
            "verification_status": preflight_verification["status"],
            "authorization_contract_id":
                preflight_verification["authorization_contract_id"],
            "authorization_expression":
                preflight_verification["authorization_expression"],
        },
        "ladder": preflight_doc["ladder"],
        "technical": preflight_doc["technical"],
        "semantic": semantic,
        "all_required_conjuncts_pass": True,
    })


def verify_final_release_attestation(
    repo: Path, final_path: str, expected_head: str,
) -> dict[str, Any]:
    head = _resolve_commit(repo, expected_head, "expected final head")
    current = str(_git(repo, "rev-parse", "HEAD")).strip()
    origin = str(_git(repo, "rev-parse", "origin/trunk")).strip()
    if current != head or origin != head:
        raise ReleaseError("final release verification requires pushed HEAD")
    if str(_git(repo, "status", "--porcelain")).strip():
        raise ReleaseError("final release worktree is not clean")
    final_doc, raw = _load_committed_json(repo, head, final_path)
    _verify_payload(final_doc, "final release attestation")
    if (final_doc.get("schema") != SCHEMA or
            final_doc.get("authorization_contract_id") != CONTRACT_ID or
            final_doc.get("status") != "PASS" or
            final_doc.get("release_state") != "SEMANTIC_RESULT_RELEASED" or
            final_doc.get("all_required_conjuncts_pass") is not True):
        raise ReleaseError("final release attestation state differs")
    release_head = _resolve_commit(
        repo, final_doc.get("release_head", ""), "attested release head")
    _require_ancestor(repo, release_head, head, "attested release head")
    recomputed = build_final_release_attestation(
        repo, preflight_path=final_doc["preflight"]["path"],
        semantic_run_dir=final_doc["semantic"]["run_dir"],
        semantic_result_commit=final_doc["semantic"]["result_commit"],
        release_head=head)
    # The final packet was created at its parent release head. Recompute its
    # evidence there logically while allowing this descendant to contain the
    # packet itself.
    recomputed["release_head"] = release_head
    recomputed["payload_sha256"] = _payload_sha256(recomputed)
    if recomputed != final_doc:
        raise ReleaseError("final release attestation differs from recomputation")
    return {
        "status": "PASS", "authorization_contract_id": CONTRACT_ID,
        "final_path": final_path,
        "final_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "final_payload_sha256": final_doc["payload_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("seal", "verify", "finalize", "verify-final"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ladder-result-commit")
    parser.add_argument("--ladder-path", default=ELIGIBLE_LADDER_PATH)
    parser.add_argument("--technical-result-commit")
    parser.add_argument("--technical-run-dir")
    parser.add_argument("--evidence-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attestation-path")
    parser.add_argument("--expected-launch-commit")
    parser.add_argument("--semantic-run-dir")
    parser.add_argument("--semantic-result-commit")
    parser.add_argument("--release-head")
    parser.add_argument("--final-path")
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.mode == "seal":
            required = (
                args.ladder_result_commit, args.technical_result_commit,
                args.technical_run_dir, args.evidence_commit, args.output)
            if any(value is None for value in required):
                parser.error("seal requires all evidence arguments and --output")
            try:
                output_relative = args.output.resolve().relative_to(repo).as_posix()
            except ValueError as exc:
                raise ReleaseError("preflight output is outside repository") from exc
            if not PREFLIGHT_PATH_RE.fullmatch(output_relative):
                raise ReleaseError(
                    "preflight output must match results/v10_release/preflight_<unique>.json")
            doc = build_release_attestation(
                repo, evidence_commit=args.evidence_commit,
                ladder_result_commit=args.ladder_result_commit,
                ladder_path=args.ladder_path,
                technical_result_commit=args.technical_result_commit,
                technical_run_dir=args.technical_run_dir)
            _atomic_write(args.output, doc)
            result = {
                "status": "PASS", "mode": "seal",
                "output": str(args.output),
                "payload_sha256": doc["payload_sha256"]}
        elif args.mode == "verify":
            if not args.attestation_path or not args.expected_launch_commit:
                parser.error(
                    "verify requires --attestation-path and --expected-launch-commit")
            result = verify_committed_attestation(
                repo, args.attestation_path, args.expected_launch_commit)
        elif args.mode == "finalize":
            required = (
                args.attestation_path, args.semantic_run_dir,
                args.semantic_result_commit, args.release_head, args.output)
            if any(value is None for value in required):
                parser.error("finalize requires preflight, semantic evidence, release head, and output")
            doc = build_final_release_attestation(
                repo, preflight_path=args.attestation_path,
                semantic_run_dir=args.semantic_run_dir,
                semantic_result_commit=args.semantic_result_commit,
                release_head=args.release_head)
            _atomic_write(args.output, doc)
            result = {
                "status": "PASS", "mode": "finalize",
                "output": str(args.output),
                "payload_sha256": doc["payload_sha256"]}
        else:
            if not args.final_path or not args.expected_launch_commit:
                parser.error(
                    "verify-final requires --final-path and --expected-launch-commit")
            result = verify_final_release_attestation(
                repo, args.final_path, args.expected_launch_commit)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
