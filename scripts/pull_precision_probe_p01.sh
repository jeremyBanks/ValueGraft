#!/usr/bin/env bash
# Receipt-driven, lossless p01 artifact pull.  A verified PASS or ERROR receipt
# is installable; an unreceipted deadline snapshot is quarantined, never lost.
set -euo pipefail

NAME="${1:?usage: pull_precision_probe_p01.sh POD_NAME}"
ROOT="${SC_REPO_ROOT:-/Users/jeb/experimentation}"
STATE="$ROOT/.pod_${NAME}_state.json"
KEY="$HOME/.ssh/id_ed25519_runpod"
REMOTE_REPO=/workspace/repo_precision_probe_p01
PACKAGER="$ROOT/scripts/historical/coherent_canary_v12_e01/treatment_packaging/artifact_packager.py"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PULL_ROOT="$ROOT/.sol-v4/precision_probe_p01/pulls/${NAME}_${STAMP}"
POD_LOGS="$ROOT/.sol-v4/precision_probe_p01/pod_logs"
cd "$ROOT"

[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "unsafe pod name" >&2; exit 2; }
[ -f "$STATE" ] || { echo "missing pod state: $STATE" >&2; exit 2; }
[ -f "$PACKAGER" ] || { echo "missing historical packager: $PACKAGER" >&2; exit 2; }
mkdir -p "$PULL_ROOT" "$POD_LOGS"

STATUS="$(SC_POD_STATE="$STATE" uv run python src/pod.py status)"
read -r IP PORT < <(python3 -c '
import json, sys
value=json.load(sys.stdin)
print(value.get("publicIp") or "", (value.get("portMappings") or {}).get("22") or "")
' <<< "$STATUS")
[ -n "$IP" ] && [ -n "$PORT" ] || { echo "pod has no SSH endpoint" >&2; exit 3; }

SSH_OPTS=(-i "$KEY" -p "$PORT" -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=20 -o ServerAliveInterval=15
          -o ServerAliveCountMax=4)
RSYNC_SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

rsync -az --partial -e "$RSYNC_SSH" \
  "root@$IP:/workspace/exp/job.log" \
  "$POD_LOGS/precision-p01_${NAME}_${STAMP}.log" 2>/dev/null || true

REMOTE_RECEIPT="$(ssh -n "${SSH_OPTS[@]}" "root@$IP" \
  'cat /workspace/exp/p01_current_receipt.txt 2>/dev/null' 2>/dev/null || true)"
if [[ ! "$REMOTE_RECEIPT" =~ ^results/precision_probe_p01/precision-probe-p01-receipt_Qwen3-30B-A3B-Instruct-2507_[0-9]{8}T[0-9]{6}Z\.json$ ]]; then
  # Deadline/abrupt failure path: make the best available result tree visible
  # in the scientific audit trail before the caller terminates the pod.  It is
  # explicitly quarantined because no receipt can authenticate completeness.
  QUARANTINE="results/_QUARANTINE_precision_probe_p01_unreceipted_Qwen3-30B-A3B-Instruct-2507_${NAME}_${STAMP}"
  mkdir -p "$QUARANTINE"
  rsync -az --partial -e "$RSYNC_SSH" \
    "root@$IP:$REMOTE_REPO/results/precision_probe_p01/" "$QUARANTINE/" \
    2>/dev/null || true
  cp "$POD_LOGS/precision-p01_${NAME}_${STAMP}.log" "$QUARANTINE/job.log" \
    2>/dev/null || true
  python3 - "$QUARANTINE" "$NAME" "$REMOTE_RECEIPT" <<'PY'
from datetime import datetime, timezone
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
value={
    "schema": "precision_probe_p01_unreceipted_quarantine_v1",
    "protocol_id": "precision-probe-p01",
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "pod_name": sys.argv[2],
    "remote_receipt_pointer_observed": sys.argv[3] or None,
    "reason": "terminal receipt absent or invalid; completeness unverified",
}
(root / "README.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY
  echo "P01 UNRECEIPTED SNAPSHOT preserved=$QUARANTINE" >&2
  exit 8
fi

STAGED_REPO="$PULL_ROOT/repo"
RECEIPT_ONLY="$PULL_ROOT/receipt.json"
FILE_LIST="$PULL_ROOT/receipt-files.txt"
mkdir -p "$STAGED_REPO"
ssh -n "${SSH_OPTS[@]}" "root@$IP" \
  "test -f $REMOTE_REPO/$REMOTE_RECEIPT"
rsync -az --partial -e "$RSYNC_SSH" \
  "root@$IP:$REMOTE_REPO/$REMOTE_RECEIPT" "$RECEIPT_ONLY"
python3 - "$RECEIPT_ONLY" "$REMOTE_RECEIPT" > "$FILE_LIST" <<'PY'
import json, re, sys
from pathlib import PurePosixPath
receipt=sys.argv[2]
doc=json.load(open(sys.argv[1]))
if (not isinstance(doc,dict) or
        doc.get("schema") != "precision_probe_p01_pod_receipt_v1" or
        doc.get("protocol_id") != "precision-probe-p01"):
    raise SystemExit("outer receipt header differs")
rows=doc.get("artifacts")
if not isinstance(rows,list) or not rows:
    raise SystemExit("outer receipt artifact inventory is absent")
paths=[]
for row in rows:
    value=row.get("path") if isinstance(row,dict) else None
    if (not isinstance(value,str) or
            not re.fullmatch(r"[A-Za-z0-9_./-]+", value)):
        raise SystemExit(f"unsafe receipt artifact path: {value!r}")
    pure=PurePosixPath(value)
    if (pure.is_absolute() or ".." in pure.parts or
            not value.startswith("results/precision_probe_p01/")):
        raise SystemExit(f"out-of-root receipt artifact path: {value}")
    paths.append(value)
if len(paths) != len(set(paths)):
    raise SystemExit("receipt artifact path is duplicated")
paths.append(receipt)
print("\n".join(paths))
PY
rsync -az --partial --files-from="$FILE_LIST" -e "$RSYNC_SSH" \
  "root@$IP:$REMOTE_REPO/" "$STAGED_REPO/"
STAGED_TREE="$STAGED_REPO/results/precision_probe_p01"
STAGED_RECEIPT="$STAGED_REPO/$REMOTE_RECEIPT"
[ -f "$STAGED_RECEIPT" ] || { echo "staged receipt is absent" >&2; exit 5; }

VALIDATION_OUTPUT="$(python3 - \
  "$STAGED_TREE" "$STAGED_RECEIPT" "$REMOTE_RECEIPT" <<'PY'
import hashlib, json, math, re, sys
from pathlib import Path, PurePosixPath

tree=Path(sys.argv[1])
receipt_path=Path(sys.argv[2])
receipt_relative=sys.argv[3]
doc=json.loads(receipt_path.read_bytes())
required={
    "schema", "protocol_id", "completed_at_utc", "expected_commit",
    "observed_commit", "model", "revision", "runner_exit",
    "runner_completion_status", "runner_completion_path",
    "packaging_exit", "lossless_package_count", "outcome_package_count",
    "recovery_raw_count",
    "terminal_status", "formal_v12_decision_eligible",
    "v12_reentry_authorized",
    "component_reuse_does_not_inherit_v12_eligibility", "provider",
    "artifacts",
}
if set(doc) != required:
    raise SystemExit(f"receipt field set differs: {sorted(set(doc) ^ required)}")
if (doc["schema"] != "precision_probe_p01_pod_receipt_v1" or
        doc["protocol_id"] != "precision-probe-p01"):
    raise SystemExit("receipt schema/protocol differs")
if (doc["expected_commit"] != doc["observed_commit"] or
        not re.fullmatch(r"[0-9a-f]{40}", doc["expected_commit"])):
    raise SystemExit("receipt commit binding differs")
if (doc["model"] != "Qwen/Qwen3-30B-A3B-Instruct-2507" or
        doc["revision"] != "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"):
    raise SystemExit("receipt subject differs")
if (doc["formal_v12_decision_eligible"] is not False or
        doc["v12_reentry_authorized"] is not False or
        doc["component_reuse_does_not_inherit_v12_eligibility"] is not True):
    raise SystemExit("receipt v12 boundary differs")
provider=doc["provider"]
if not isinstance(provider, dict):
    raise SystemExit("provider receipt is absent")
if (not isinstance(provider.get("pod_name"),str) or
        not provider.get("pod_name") or
        not isinstance(provider.get("provider_pod_id"),str) or
        not provider.get("provider_pod_id")):
    raise SystemExit("provider identity is absent")
rate=provider.get("hourly_cost_usd")
cap=provider.get("provider_wall_cap_seconds")
scientific_cap=provider.get("scientific_extension_cap_seconds")
interrupt_elapsed=provider.get("operational_interrupt_elapsed_seconds")
interrupt_epoch=provider.get("operational_interrupt_epoch")
if (isinstance(rate, bool) or not isinstance(rate, (int,float)) or
        not math.isfinite(rate) or rate <= 0 or
        isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 7200 or
        scientific_cap != cap or
        isinstance(interrupt_elapsed, bool) or not isinstance(interrupt_elapsed, int) or
        interrupt_elapsed != cap - 180 or interrupt_elapsed <= 0 or
        isinstance(interrupt_epoch, bool) or not isinstance(interrupt_epoch, int) or
        interrupt_epoch != provider.get("provider_clock_started_epoch") + interrupt_elapsed or
        cap * float(rate) / 3600 > 4.0 + 1e-9 or
        provider.get("max_cost_usd") != 4.0):
    raise SystemExit("provider cap differs")

artifacts=doc["artifacts"]
if not isinstance(artifacts, list) or not artifacts:
    raise SystemExit("receipt artifact inventory is empty")
expected=set()
completion=[]
for index,row in enumerate(artifacts):
    if not isinstance(row,dict) or set(row) != {"path","sha256","size_bytes"}:
        raise SystemExit(f"artifact row {index} differs")
    relative=row["path"]
    pure=PurePosixPath(relative)
    if (not isinstance(relative,str) or pure.is_absolute() or ".." in pure.parts or
            not relative.startswith("results/precision_probe_p01/") or
            not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or
            isinstance(row["size_bytes"],bool) or
            not isinstance(row["size_bytes"],int) or row["size_bytes"] <= 0):
        raise SystemExit(f"unsafe artifact row {index}")
    if relative in expected:
        raise SystemExit(f"duplicate artifact: {relative}")
    expected.add(relative)
    local=tree / PurePosixPath(relative).relative_to("results/precision_probe_p01")
    if not local.is_file() or local.is_symlink():
        raise SystemExit(f"artifact is absent/not regular: {relative}")
    data=local.read_bytes()
    if len(data) != row["size_bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise SystemExit(f"artifact differs from receipt: {relative}")
    if re.fullmatch(
        r"precision-probe-p01-completion_Qwen3-30B-A3B-Instruct-2507_"
        r"[0-9]{8}T[0-9]{6}(?:[0-9]{6})?Z\.json", pure.name):
        completion.append(local)

actual={
    "results/precision_probe_p01/" + path.relative_to(tree).as_posix()
    for path in tree.rglob("*") if path.is_file() and path != receipt_path
}
if actual != expected:
    raise SystemExit(
        f"staged inventory differs; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
if receipt_relative != "results/precision_probe_p01/" + receipt_path.name:
    raise SystemExit("receipt pointer/path differs")
terminal=doc["terminal_status"]
packages=doc["lossless_package_count"]
outcome_packages=doc["outcome_package_count"]
recovery=doc["recovery_raw_count"]
if (terminal not in {"PASS","ERROR"} or
        any(isinstance(value,bool) or not isinstance(value,int) or value < 0
            for value in (packages,outcome_packages,recovery))):
    raise SystemExit("outer terminal/count fields differ")

# ERROR is still a terminal, receipt-bound scientific artifact.  It may have
# no runner completion marker (hard timeout/setup failure) or an adverse one.
# Outer inventory/hash integrity is sufficient to install that partial tree;
# only PASS requires the full runner/gate reconstruction below.
if terminal == "ERROR":
    if len(completion) > 1:
        raise SystemExit("ERROR receipt binds multiple runner completions")
    expected_completion_path=(
        "results/precision_probe_p01/" + completion[0].relative_to(tree).as_posix()
        if completion else None)
    if doc["runner_completion_path"] != expected_completion_path:
        raise SystemExit("ERROR outer completion path differs")
    if (doc["runner_completion_status"] is not None and
            not isinstance(doc["runner_completion_status"],str)):
        raise SystemExit("ERROR outer completion status differs")
    print(terminal, packages, recovery)
    raise SystemExit(0)

if len(completion) != 1:
    raise SystemExit(f"expected one runner completion, found {len(completion)}")
run_completion=json.loads(completion[0].read_bytes())
expected_completion_fields = {
    "schema", "protocol_id", "formal_v12_decision_eligible",
    "v12_reentry_authorized",
    "component_reuse_does_not_inherit_v12_eligibility",
    "semantic_evidence_eligible", "status", "model", "revision",
    "created_at_utc", "run_manifest", "artifact_inventory",
    "recovery_raw_files", "provider_elapsed_seconds",
    "estimated_provider_cost_usd", "matched_runtime_gate",
}
if set(run_completion) != expected_completion_fields:
    raise SystemExit("runner completion field set differs")
if (run_completion.get("schema") != "precision_probe_p01_completion_v1" or
        run_completion.get("protocol_id") != "precision-probe-p01" or
        run_completion.get("formal_v12_decision_eligible") is not False or
        run_completion.get("v12_reentry_authorized") is not False or
        run_completion.get(
            "component_reuse_does_not_inherit_v12_eligibility") is not True or
        run_completion.get("semantic_evidence_eligible") is not False or
        run_completion.get("model") != "Qwen/Qwen3-30B-A3B-Instruct-2507" or
        run_completion.get("revision") !=
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"):
    raise SystemExit("runner completion identity/boundary differs")
if (doc["runner_completion_status"] != run_completion.get("status") or
        doc["runner_completion_path"] !=
            "results/precision_probe_p01/" + completion[0].relative_to(tree).as_posix()):
    raise SystemExit("outer receipt completion binding differs")

def resolve_runner_path(value):
    if not isinstance(value, str):
        raise SystemExit("runner binding path is not a string")
    pure=PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts:
        raise SystemExit(f"unsafe runner binding path: {value}")
    if value.startswith("results/precision_probe_p01/"):
        return tree / pure.relative_to("results/precision_probe_p01")
    return completion[0].parent / Path(*pure.parts)

def verify_runner_binding(binding):
    if not isinstance(binding,dict) or set(binding) != {"path","sha256","size_bytes"}:
        raise SystemExit("runner binding field set differs")
    path=resolve_runner_path(binding["path"])
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"runner-bound artifact is absent: {path}")
    data=path.read_bytes()
    if (len(data) != binding["size_bytes"] or
            hashlib.sha256(data).hexdigest() != binding["sha256"]):
        raise SystemExit(f"runner-bound artifact differs: {path}")
    return path

manifest_path=verify_runner_binding(run_completion["run_manifest"])
runner_inventory=run_completion["artifact_inventory"]
if not isinstance(runner_inventory,list) or not runner_inventory:
    raise SystemExit("runner completion inventory is empty")
runner_paths=[verify_runner_binding(binding) for binding in runner_inventory]
if len(runner_paths) != len(set(runner_paths)):
    raise SystemExit("runner completion inventory is duplicated")
actual_runner_paths={
    path for path in completion[0].parent.rglob("*")
    if path.is_file() and path != completion[0]
}
if set(runner_paths) != actual_runner_paths:
    raise SystemExit("runner completion inventory is not exhaustive")
manifest=json.loads(manifest_path.read_bytes())
regimes=manifest.get("regimes")
if not isinstance(regimes,dict) or set(regimes) != {"nf4","bf16"}:
    raise SystemExit("runner manifest regime set differs")

def visit_outcomes(value):
    count=0
    valid=True
    if isinstance(value,dict):
        if "completion_status" in value:
            if value.get("completion_status") != "COMPLETE":
                valid=False
            else:
                package=value.get("raw_package")
                if (not isinstance(package,dict) or
                        package.get("verification_status") != "VERIFIED"):
                    valid=False
                else:
                    count += 1
        for child in value.values():
            child_count, child_valid = visit_outcomes(child)
            count += child_count
            valid = valid and child_valid
    elif isinstance(value,list):
        for child in value:
            child_count, child_valid = visit_outcomes(child)
            count += child_count
            valid = valid and child_valid
    return count, valid

outcome_count=0
runner_gates_ok=True
for regime_name in ("nf4","bf16"):
    regime=regimes[regime_name]
    technical=regime.get("technical") if isinstance(regime,dict) else None
    runner_gates_ok = runner_gates_ok and (
        isinstance(regime,dict) and
        regime.get("status") == "OUTCOMES_FINISHED" and
        isinstance(technical,dict) and
        technical.get("status") == "PASS" and
        technical.get("regime") == regime_name)
    regime_count, regime_valid = visit_outcomes(regime)
    outcome_count += regime_count
    runner_gates_ok = runner_gates_ok and regime_valid

completion_elapsed=run_completion["provider_elapsed_seconds"]
completion_cost=run_completion["estimated_provider_cost_usd"]
completion_ok=(
    run_completion.get("status") == "COMPLETE" and
    run_completion.get("recovery_raw_files") == [] and runner_gates_ok and
    isinstance(run_completion.get("matched_runtime_gate"),dict) and
    run_completion["matched_runtime_gate"].get("status") == "PASS" and
    outcome_count >= 4 and
    isinstance(completion_elapsed,(int,float)) and
    not isinstance(completion_elapsed,bool) and
    math.isfinite(float(completion_elapsed)) and
    0 <= completion_elapsed <= cap and
    isinstance(completion_cost,(int,float)) and not isinstance(completion_cost,bool) and
    math.isfinite(float(completion_cost)) and 0 <= completion_cost <= 4.0)
pass_contract=(
    doc["runner_exit"] == 0 and doc["packaging_exit"] == 0 and
    isinstance(packages,int) and not isinstance(packages,bool) and packages >= 6 and
    isinstance(outcome_packages,int) and not isinstance(outcome_packages,bool) and
    outcome_packages >= 4 and
    recovery == 0 and completion_ok and
    provider.get("elapsed_seconds_at_receipt", cap + 1) <= cap
)
if terminal not in {"PASS","ERROR"} or ((terminal == "PASS") != pass_contract):
    raise SystemExit("receipt terminal status differs from reconstructed contract")
print(terminal, packages, recovery)
PY
)"
read -r TERMINAL_STATUS PACKAGE_COUNT RECOVERY_COUNT <<< "$VALIDATION_OUTPUT"

# Verify every historical package itself after transport, not only its outer
# receipt hashes.  ERROR trees may legitimately contain recovery raw instead.
OBSERVED_PACKAGES=0
PACKAGE_VERIFY_FAILURES=0
while IFS= read -r manifest; do
  [ -n "$manifest" ] || continue
  OBSERVED_PACKAGES=$((OBSERVED_PACKAGES + 1))
  if ! python3 "$PACKAGER" verify "$(dirname "$manifest")"; then
    PACKAGE_VERIFY_FAILURES=$((PACKAGE_VERIFY_FAILURES + 1))
  fi
done < <(find "$STAGED_TREE" -type f -path '*.lossless-package/manifest.json' -print | sort)
if [ "$TERMINAL_STATUS" = "PASS" ] && { \
    [ "$OBSERVED_PACKAGES" -ne "$PACKAGE_COUNT" ] || \
    [ "$PACKAGE_VERIFY_FAILURES" -ne 0 ]; }; then
  echo "PASS package verification differs: observed=$OBSERVED_PACKAGES receipt=$PACKAGE_COUNT failures=$PACKAGE_VERIFY_FAILURES" >&2
  exit 6
fi

# Install only after the complete tree and every package verify.  Existing
# identical files are idempotent; any differing destination fails closed.
python3 - "$STAGED_TREE" "$STAGED_RECEIPT" "$REMOTE_RECEIPT" "$ROOT" <<'PY'
import hashlib, json, os, shutil, sys
from pathlib import Path, PurePosixPath
tree, receipt_source, receipt_relative, repo = (
    Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
doc=json.loads(receipt_source.read_bytes())

def digest(path):
    data=path.read_bytes()
    return len(data), hashlib.sha256(data).hexdigest()

def install(source, destination):
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or digest(source) != digest(destination):
            raise SystemExit(f"refusing different existing destination: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd=os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd,"wb") as out, source.open("rb") as inp:
            fd=-1
            shutil.copyfileobj(inp,out,1 << 20)
            out.flush(); os.fsync(out.fileno())
    except Exception:
        if fd >= 0: os.close(fd)
        raise

for row in doc["artifacts"]:
    relative=PurePosixPath(row["path"])
    source=tree / relative.relative_to("results/precision_probe_p01")
    install(source, repo / Path(*relative.parts))
install(receipt_source, repo / Path(*PurePosixPath(receipt_relative).parts))
PY

echo "P01 PULL VERIFIED receipt=$REMOTE_RECEIPT terminal_status=$TERMINAL_STATUS packages=$PACKAGE_COUNT package_verify_failures=$PACKAGE_VERIFY_FAILURES recovery_raw=$RECOVERY_COUNT"
git status --short -- results/precision_probe_p01
