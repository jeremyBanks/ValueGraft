"""Terminal integrity and two-process authorization for coherent-state v9.

This module deliberately contains no model execution.  It seals terminal
apparatus payloads, inventories the complete executable apparatus, and verifies
that a prior technical result is both byte-for-byte committed and independently
harvest-valid before a separate semantic process may load the subject.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


SCHEMA = 2
DESIGN_ID = "coherent-state-gapped-v9"
AMENDMENT_ID = "COHERENT-STATE-PREREGISTRATION-AMENDMENTS-1-2-3-4-5-6-7-8-9"
INDEX_NAME = "terminal_artifact_index.json"
RECEIPT_NAME = "terminal_receipt.json"

# Exact roots plus deliberately narrow globs.  The aggregate includes every
# matched path and path name, so adding, removing, or changing an apparatus file
# invalidates a prior technical authorization.
APPARATUS_REQUIRED = (
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-1.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-2.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-3.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-4.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-5.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-6.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-7.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-8.md",
    "COHERENT-STATE-PREREGISTRATION-AMENDMENT-9.md",
    "src/analyze_coherent_state.py",
    "src/arms_common.py",
    "src/coherent_state_calibration.py",
    "src/coherent_state_cases.py",
    "src/coherent_state_hf.py",
    "src/coherent_state_integrity.py",
    "src/coherent_state_runtime.py",
    "src/coherent_state_store.py",
    "src/coherent_state_tokens.py",
    "src/cross_arch_probe.py",
    "src/l_coherent_state_hf.py",
    "src/kvlib_hf.py",
    "src/pod.py",
    "src/run_coherent_state_hf.py",
    "src/validate_coherent_external_donors.py",
    "scripts/classify_pod.sh",
    "scripts/coherent_lifecycle_lib.sh",
    "scripts/coherent_monitor_selftest.sh",
    "scripts/job_coherent_state_bf16.sh",
    "scripts/job_coherent_state_semantic_bf16.sh",
    "scripts/launch_pod.sh",
    "scripts/preflight.py",
    "scripts/preflight.sh",
    "scripts/validate_coherent_harvest.py",
    "scripts/watch_coherent_state_pod.sh",
)
APPARATUS_GLOBS = (
    "src/coherent_state_*.py",
    "scripts/*coherent*.sh",
    "scripts/*coherent*.py",
)


class IntegrityError(RuntimeError):
    """Fail-closed integrity or authorization error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(doc: dict[str, Any]) -> str:
    unhashed = {key: value for key, value in doc.items()
                if key != "payload_sha256"}
    return hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()


def _normalize_terminal_json(value: Any) -> Any:
    """Make non-finite failed measurements explicit and canonical JSON-safe."""
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _normalize_terminal_json(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_terminal_json(item) for item in value]
    return value


def seal_payload(doc: dict[str, Any]) -> dict[str, Any]:
    sealed = _normalize_terminal_json({
        key: value for key, value in doc.items() if key != "payload_sha256"})
    sealed["payload_sha256"] = payload_sha256(sealed)
    return sealed


def verify_payload(doc: dict[str, Any], label: str) -> None:
    observed = doc.get("payload_sha256")
    expected = payload_sha256(doc)
    if observed != expected:
        raise IntegrityError(
            f"{label} payload_sha256 {observed!r} != recomputed {expected}")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def write_sealed_payload(path: Path, doc: dict[str, Any]) -> dict[str, Any]:
    sealed = seal_payload(doc)
    _atomic_bytes(path, canonical_json_bytes(sealed) + b"\n")
    reread = json.loads(path.read_text())
    verify_payload(reread, str(path))
    if reread != sealed:
        raise IntegrityError(f"{path} changed during sealed write")
    return sealed


def apparatus_inventory(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    selected = set(APPARATUS_REQUIRED)
    for pattern in APPARATUS_GLOBS:
        selected.update(
            path.relative_to(repo).as_posix()
            for path in repo.glob(pattern) if path.is_file())
    missing = [relative for relative in APPARATUS_REQUIRED
               if not (repo / relative).is_file()]
    if missing:
        raise IntegrityError(f"required apparatus files absent: {missing}")
    records = []
    for relative in sorted(selected):
        path = repo / relative
        if not path.is_file():
            continue
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": raw_sha256(path),
        })
    aggregate = hashlib.sha256(canonical_json_bytes(records)).hexdigest()
    return {
        "files": records,
        "file_count": len(records),
        "aggregate_sha256": aggregate,
    }


def _relative_payload_paths(run_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*.json")
        if path.name not in {INDEX_NAME, RECEIPT_NAME}
    )


def write_terminal_envelope(
    run_dir: Path,
    required_payload_paths: Iterable[str],
    *,
    terminal_status: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Index every terminal JSON payload, then durably receipt the index.

    Payload files must already be immutable and sealed.  An existing envelope
    is never overwritten; terminal attempts are immutable.
    """
    index_path = run_dir / INDEX_NAME
    receipt_path = run_dir / RECEIPT_NAME
    if index_path.exists() or receipt_path.exists():
        raise IntegrityError("terminal envelope already exists")
    required = sorted(set(required_payload_paths))
    observed = _relative_payload_paths(run_dir)
    if observed != required:
        raise IntegrityError(
            f"terminal payload path set {observed} != required {required}")
    rows = []
    for relative in required:
        path = run_dir / relative
        doc = json.loads(path.read_text())
        if not isinstance(doc, dict):
            raise IntegrityError(f"terminal payload is not an object: {relative}")
        verify_payload(doc, relative)
        rows.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "raw_sha256": raw_sha256(path),
            "payload_sha256": doc["payload_sha256"],
        })
    index = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "amendment_id": AMENDMENT_ID,
        "status": terminal_status,
        "required_apparatus_payload_paths": required,
        "artifacts": rows,
    }
    _atomic_bytes(index_path, canonical_json_bytes(index) + b"\n")
    if json.loads(index_path.read_text()) != index:
        raise IntegrityError("terminal artifact index read-back mismatch")
    receipt = {
        "schema": SCHEMA,
        "design_id": DESIGN_ID,
        "amendment_id": AMENDMENT_ID,
        "status": terminal_status,
        "index_path": INDEX_NAME,
        "index_bytes": index_path.stat().st_size,
        "index_raw_sha256": raw_sha256(index_path),
    }
    _atomic_bytes(receipt_path, canonical_json_bytes(receipt) + b"\n")
    if json.loads(receipt_path.read_text()) != receipt:
        raise IntegrityError("terminal receipt read-back mismatch")
    verify_terminal_envelope(run_dir)
    return index, receipt


def verify_terminal_envelope(run_dir: Path) -> dict[str, Any]:
    index_path = run_dir / INDEX_NAME
    receipt_path = run_dir / RECEIPT_NAME
    try:
        index = json.loads(index_path.read_text())
        receipt = json.loads(receipt_path.read_text())
    except Exception as exc:
        raise IntegrityError(f"terminal envelope unreadable: {exc}") from exc
    for label, doc in (("index", index), ("receipt", receipt)):
        if (doc.get("schema") != SCHEMA or doc.get("design_id") != DESIGN_ID or
                doc.get("amendment_id") != AMENDMENT_ID or
                doc.get("status") not in {"PASS", "FAIL"}):
            raise IntegrityError(f"terminal {label} identity/status mismatch")
    if index["status"] != receipt["status"]:
        raise IntegrityError("terminal index and receipt statuses differ")
    if receipt.get("index_path") != INDEX_NAME:
        raise IntegrityError("terminal receipt names the wrong index")
    if receipt.get("index_bytes") != index_path.stat().st_size:
        raise IntegrityError("terminal receipt index byte count mismatch")
    if receipt.get("index_raw_sha256") != raw_sha256(index_path):
        raise IntegrityError("terminal receipt index hash mismatch")
    required = index.get("required_apparatus_payload_paths")
    if not isinstance(required, list) or required != sorted(set(required)):
        raise IntegrityError("terminal index required path set is malformed")
    if _relative_payload_paths(run_dir) != required:
        raise IntegrityError("terminal directory has missing or unindexed JSON")
    rows = index.get("artifacts")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != required:
        raise IntegrityError("terminal index artifact rows do not match path set")
    for row in rows:
        path = run_dir / row["path"]
        if row.get("bytes") != path.stat().st_size:
            raise IntegrityError(f"indexed byte count mismatch: {row['path']}")
        if row.get("raw_sha256") != raw_sha256(path):
            raise IntegrityError(f"indexed raw hash mismatch: {row['path']}")
        doc = json.loads(path.read_text())
        verify_payload(doc, row["path"])
        if row.get("payload_sha256") != doc.get("payload_sha256"):
            raise IntegrityError(f"indexed payload hash mismatch: {row['path']}")
    return {"index": index, "receipt": receipt}


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout


def verify_committed_directory(
    repo: Path, run_dir: Path, result_commit: str, launch_commit: str,
) -> dict[str, Any]:
    repo = repo.resolve()
    run_dir = run_dir.resolve()
    try:
        relative_root = run_dir.relative_to(repo).as_posix()
    except ValueError as exc:
        raise IntegrityError("technical authorization directory is outside repo") from exc
    resolved_result = str(_git(
        repo, "rev-parse", f"{result_commit}^{{commit}}" )).strip()
    resolved_launch = str(_git(
        repo, "rev-parse", f"{launch_commit}^{{commit}}" )).strip()
    branch = str(_git(repo, "symbolic-ref", "--short", "HEAD")).strip()
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if branch != "trunk" or resolved_launch != head:
        raise IntegrityError("semantic authorization must launch from trunk HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_result, resolved_launch],
        cwd=repo,
    ).returncode == 0
    if not ancestor:
        raise IntegrityError("technical result commit is not an ancestor of launch")
    on_trunk = subprocess.run(
        ["git", "merge-base", "--is-ancestor", resolved_result,
         "refs/heads/trunk"], cwd=repo,
    ).returncode == 0
    if not on_trunk:
        raise IntegrityError("technical result commit is not on local trunk")
    tree_output = str(_git(
        repo, "ls-tree", "-r", "--name-only", resolved_result, "--", relative_root))
    tree_paths = sorted(row for row in tree_output.splitlines() if row)
    disk_paths = sorted(
        path.relative_to(repo).as_posix()
        for path in run_dir.rglob("*") if path.is_file())
    if tree_paths != disk_paths:
        raise IntegrityError(
            f"committed technical directory paths differ: tree={tree_paths}, "
            f"disk={disk_paths}")
    rows = []
    for relative in tree_paths:
        committed = bytes(_git(
            repo, "show", f"{resolved_result}:{relative}", binary=True))
        disk = (repo / relative).read_bytes()
        if committed != disk:
            raise IntegrityError(f"committed bytes differ for {relative}")
        rows.append({
            "path": relative,
            "bytes": len(disk),
            "sha256": hashlib.sha256(disk).hexdigest(),
        })
    return {
        "result_commit": resolved_result,
        "semantic_launch_commit": resolved_launch,
        "run_dir": relative_root,
        "files": rows,
    }


def verify_harvest_attestation(
    repo: Path, run_dir: Path, result_commit: str,
) -> dict[str, Any]:
    """Verify the committed sibling harvest payload and re-run its validator."""
    repo = repo.resolve()
    sibling = Path(f"{run_dir}.harvest_validation.json").resolve()
    try:
        sibling_relative = sibling.relative_to(repo).as_posix()
        run_relative = run_dir.resolve().relative_to(repo).as_posix()
    except ValueError as exc:
        raise IntegrityError("harvest authorization paths are outside repo") from exc
    if not sibling.is_file():
        raise IntegrityError(f"committed harvest attestation absent: {sibling}")
    committed = bytes(_git(repo, "show", f"{result_commit}:{sibling_relative}",
                           binary=True))
    if committed != sibling.read_bytes():
        raise IntegrityError("harvest attestation bytes differ from result commit")
    doc = json.loads(sibling.read_text())
    verify_payload(doc, sibling_relative)
    if doc.get("status") != "PASS" or doc.get("mode") != "technical":
        raise IntegrityError("harvest attestation is not a technical PASS")
    if doc.get("run_dir") != run_relative:
        raise IntegrityError("harvest attestation names a different run directory")
    envelope = verify_terminal_envelope(run_dir)
    expected_raw = {
        "gate": raw_sha256(run_dir / "production_kernel_gate.json"),
        "manifest": raw_sha256(run_dir / "manifest.json"),
        "index": raw_sha256(run_dir / INDEX_NAME),
        "receipt": raw_sha256(run_dir / RECEIPT_NAME),
    }
    if doc.get("raw_sha256") != expected_raw:
        raise IntegrityError("harvest attestation raw hashes do not match terminal files")
    validator_path = repo / "scripts/validate_coherent_harvest.py"
    validator = doc.get("validator") or {}
    if (validator.get("path") != "scripts/validate_coherent_harvest.py" or
            validator.get("sha256") != raw_sha256(validator_path)):
        raise IntegrityError("harvest validator identity mismatch")
    proc = subprocess.run(
        [sys.executable, str(validator_path), str(run_dir), "technical",
         "--read-only"],
        cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise IntegrityError(
            f"independent harvest revalidation failed: {proc.stderr.strip()}")
    try:
        rerun = json.loads(proc.stdout)
    except Exception as exc:
        raise IntegrityError("harvest validator did not emit JSON") from exc
    if rerun.get("status") != "PASS" or rerun.get("mode") != "technical":
        raise IntegrityError("harvest validator output is not technical PASS")
    return {
        "path": sibling_relative,
        "raw_sha256": raw_sha256(sibling),
        "payload_sha256": doc["payload_sha256"],
        "attestation": doc,
        "independent_revalidation": rerun,
        "index": envelope["index"],
        "receipt": envelope["receipt"],
    }


@dataclass(frozen=True)
class PriorTechnicalAuthorization:
    result_commit: str
    launch_commit: str
    run_dir: str
    gate_payload_sha256: str
    raw_sha256: dict[str, str]
    apparatus_inventory: dict[str, Any]
    static_fingerprint: dict[str, Any]
    committed_directory: dict[str, Any]
    harvest: dict[str, Any]


def verify_prior_technical_authorization(
    repo: Path,
    run_dir: Path,
    result_commit: str,
    launch_commit: str,
    current_apparatus: dict[str, Any],
) -> PriorTechnicalAuthorization:
    committed = verify_committed_directory(
        repo, run_dir, result_commit, launch_commit)
    envelope = verify_terminal_envelope(run_dir)
    gate_path = run_dir / "production_kernel_gate.json"
    manifest_path = run_dir / "manifest.json"
    gate = json.loads(gate_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    verify_payload(gate, "production_kernel_gate.json")
    verify_payload(manifest, "manifest.json")
    for label, doc in (("gate", gate), ("manifest", manifest)):
        if (doc.get("schema") != SCHEMA or doc.get("design_id") != DESIGN_ID or
                doc.get("amendment_id") != AMENDMENT_ID):
            raise IntegrityError(f"prior technical {label} identity mismatch")
        if (doc.get("model") != "Qwen/Qwen3-30B-A3B-Instruct-2507" or
                doc.get("revision") !=
                "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" or
                doc.get("dtype") != "torch.bfloat16" or
                doc.get("attention_backend") != "eager"):
            raise IntegrityError(f"prior technical {label} subject mismatch")
    if gate.get("status") != "PASS" or gate.get("gates", {}).get("passes") is not True:
        raise IntegrityError("prior technical gate is not terminal PASS")
    if (manifest.get("status") != "TECHNICAL_PASS" or
            manifest.get("phase") != "TECHNICAL_COMPLETE"):
        raise IntegrityError("prior technical manifest is not TECHNICAL_COMPLETE")
    if gate.get("technical_only") is not True or manifest.get("technical_only") is not True:
        raise IntegrityError("prior technical result was not technical-only")
    if manifest.get("production_kernel_gate_payload_sha256") != \
            gate.get("payload_sha256"):
        raise IntegrityError("manifest/gate payload binding differs")
    for field in (
            "fingerprint_static", "apparatus_inventory", "model", "revision",
            "dtype", "attention_backend", "attention_backend_fingerprint",
            "geometry", "context_limit"):
        if manifest.get(field) != gate.get(field):
            raise IntegrityError(f"manifest/gate {field} binding differs")
    unique_name = manifest.get("production_kernel_gate_attempt_path")
    if (not isinstance(unique_name, str) or Path(unique_name).name != unique_name or
            not unique_name.startswith("production_kernel_gate_") or
            unique_name == "production_kernel_gate.json"):
        raise IntegrityError("technical manifest lacks unique gate path")
    unique_path = run_dir / unique_name
    if unique_path.read_bytes() != gate_path.read_bytes():
        raise IntegrityError("canonical and unique gate payloads are not identical")
    stage_refs = (gate.get("gates") or {}).get("stage_refs")
    expected_stages = {
        "committed_case_schedule_fixtures",
        "external_donor_construction",
    }
    if not isinstance(stage_refs, dict) or set(stage_refs) != expected_stages:
        raise IntegrityError("technical gate heavy-stage references are not exact")
    sidecar_paths = []
    for stage_name in sorted(expected_stages):
        ref = stage_refs[stage_name]
        relative = ref.get("path") if isinstance(ref, dict) else None
        if (not isinstance(relative, str) or Path(relative).name != relative or
                not relative.startswith("technical_stage_")):
            raise IntegrityError(f"invalid heavy-stage reference: {stage_name}")
        path = run_dir / relative
        sidecar = json.loads(path.read_text())
        verify_payload(sidecar, relative)
        if (sidecar.get("schema") != SCHEMA or
                sidecar.get("design_id") != DESIGN_ID or
                sidecar.get("amendment_id") != AMENDMENT_ID or
                sidecar.get("status") != "PASS" or
                sidecar.get("stage_name") != stage_name or
                (sidecar.get("lifecycle") or {}).get("status") != "PASS"):
            raise IntegrityError(f"heavy-stage sidecar invalid: {stage_name}")
        if (ref.get("byte_count") != path.stat().st_size or
                ref.get("raw_file_sha256") != raw_sha256(path) or
                ref.get("payload_sha256") != sidecar.get("payload_sha256")):
            raise IntegrityError(f"heavy-stage sidecar binding changed: {stage_name}")
        sidecar_paths.append(relative)
    expected_paths = sorted([
        "manifest.json", "production_kernel_gate.json", unique_name,
        *sidecar_paths])
    if envelope["index"].get("required_apparatus_payload_paths") != expected_paths:
        raise IntegrityError("technical PASS terminal payload set is not exact")
    prior_apparatus = manifest.get("apparatus_inventory")
    if prior_apparatus != current_apparatus:
        raise IntegrityError("current apparatus differs from technical PASS")
    static = manifest.get("fingerprint_static") or {}
    technical_launch = static.get("code_commit")
    if not isinstance(technical_launch, str):
        raise IntegrityError("technical PASS lacks launch code commit")
    try:
        resolved_technical_launch = str(_git(
            repo, "rev-parse", f"{technical_launch}^{{commit}}" )).strip()
    except subprocess.CalledProcessError as exc:
        raise IntegrityError("technical launch commit is not resolvable") from exc
    if subprocess.run(
            ["git", "merge-base", "--is-ancestor", resolved_technical_launch,
             committed["result_commit"]], cwd=repo).returncode != 0:
        raise IntegrityError("technical launch commit is not an ancestor of result")
    for row in current_apparatus.get("files", []):
        relative = row.get("path")
        if not isinstance(relative, str):
            raise IntegrityError("apparatus inventory contains invalid path")
        try:
            launched = bytes(_git(
                repo, "show", f"{resolved_technical_launch}:{relative}",
                binary=True))
        except subprocess.CalledProcessError as exc:
            raise IntegrityError(
                f"apparatus file absent from technical launch: {relative}") from exc
        if (len(launched) != row.get("bytes") or
                hashlib.sha256(launched).hexdigest() != row.get("sha256")):
            raise IntegrityError(
                f"apparatus file changed since technical launch: {relative}")
    harvest = verify_harvest_attestation(repo, run_dir, result_commit)
    raw = {
        name: raw_sha256(run_dir / name)
        for name in (unique_name, "production_kernel_gate.json", "manifest.json",
                     INDEX_NAME, RECEIPT_NAME)
    }
    return PriorTechnicalAuthorization(
        result_commit=committed["result_commit"],
        launch_commit=committed["semantic_launch_commit"],
        run_dir=committed["run_dir"],
        gate_payload_sha256=gate["payload_sha256"],
        raw_sha256=raw,
        apparatus_inventory=current_apparatus,
        static_fingerprint=static or (manifest.get("fingerprint") or {}),
        committed_directory=committed,
        harvest=harvest,
    )
