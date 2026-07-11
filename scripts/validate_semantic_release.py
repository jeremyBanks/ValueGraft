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
import json
from pathlib import Path
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
    "scripts/validate_semantic_release.py",
    "scripts/launch_semantic_release.sh",
)
MAX_ATTESTATION_BYTES = 4_000_000


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


def verify_apparatus_bridge(repo: Path, current_apparatus: dict[str, Any]) \
        -> dict[str, Any]:
    launch = _resolve_commit(repo, LADDER_LAUNCH_COMMIT, "ladder launch")
    files = current_apparatus.get("files")
    if not isinstance(files, list) or not files:
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
    if not isinstance(artifact_rows, list):
        raise ReleaseError("ladder artifact file inventory is absent")
    artifact_by_path = {
        row.get("path"): row for row in artifact_rows if isinstance(row, dict)
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
    ladder_result_commit: str,
    ladder_path: str,
    technical_result_commit: str,
    technical_run_dir: str,
) -> dict[str, Any]:
    evidence = _resolve_commit(repo, evidence_commit, "release evidence")
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    _require_ancestor(repo, evidence, head, "release evidence")

    sys.path.insert(0, str(repo / "src"))
    from coherent_state_integrity import (  # pylint: disable=import-outside-toplevel
        apparatus_inventory, verify_prior_technical_authorization)

    apparatus = apparatus_inventory(repo)
    ladder = validate_ladder_commit(
        repo, result_commit=ladder_result_commit, ladder_path=ladder_path,
        semantic_launch_commit=evidence, current_apparatus=apparatus)
    technical = verify_prior_technical_authorization(
        repo, repo / technical_run_dir, technical_result_commit,
        evidence, apparatus)
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
    return {
        "status": "PASS",
        "authorization_contract_id": CONTRACT_ID,
        "launch_commit": launch,
        "attestation_path": attestation_path,
        "attestation_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "attestation_payload_sha256": attestation["payload_sha256"],
        "authorization_expression": "L AND T",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("seal", "verify"))
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--ladder-result-commit")
    parser.add_argument("--ladder-path", default=ELIGIBLE_LADDER_PATH)
    parser.add_argument("--technical-result-commit")
    parser.add_argument("--technical-run-dir")
    parser.add_argument("--evidence-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--attestation-path")
    parser.add_argument("--expected-launch-commit")
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        if args.mode == "seal":
            required = (
                args.ladder_result_commit, args.technical_result_commit,
                args.technical_run_dir, args.evidence_commit, args.output)
            if any(value is None for value in required):
                parser.error("seal requires all evidence arguments and --output")
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
        else:
            if not args.attestation_path or not args.expected_launch_commit:
                parser.error(
                    "verify requires --attestation-path and --expected-launch-commit")
            result = verify_committed_attestation(
                repo, args.attestation_path, args.expected_launch_commit)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL", "error_type": type(exc).__name__,
            "error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
