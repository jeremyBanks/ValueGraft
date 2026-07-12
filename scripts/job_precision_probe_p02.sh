#!/usr/bin/env bash
# Matched bf16/NF4 precision-probe p02 job.  The local launcher owns the
# provider clock; this job owns the exact checkout, isolated runtime, runner,
# lossless packaging, and terminal receipt.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo_precision_probe_p02
VENV=/workspace/p02-venv
OUTPUT_ROOT=results/precision_probe_p02
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL_SLUG="Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PREREG="PRECISION-PROBE-P02-PREREGISTRATION.md"
PREREG_SHA256="dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d"
PACKAGER="scripts/historical/coherent_canary_v12_e01/treatment_packaging/artifact_packager.py"
BUDGET_FILE="$BOOT/p02_provider_budget.json"
POINTER="$BOOT/p02_current_receipt.txt"
STAMP="$(python3 -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))')"
CLONE_TMP="/workspace/repo_precision_probe_p02_${STAMP}.tmp"
# Stop model-facing work 210 seconds before C.  GNU timeout's 60-second kill
# grace then ends by C-150, leaving 30 seconds for the outer shell to verify
# packages and publish its receipt before the forced pull begins at C-120.
RUNNER_INTERRUPT_LEAD_SECONDS=210

echo "START PRECISION_PROBE_P02 $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION"
echo "PROTOCOL=precision-probe-p02 ORDER=NF4-then-bf16 ATTENTION=eager KV_DTYPE=bfloat16"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader,nounits

for token_file in "$BOOT/.hf_key" "$BOOT/.huggingface_key"; do
  if [ -s "$token_file" ]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$token_file")"
    break
  fi
done
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF token absent"; exit 2; }

# launch_pod.sh starts this detached job before the p02 wrapper can bind the
# returned provider rate.  Wait only for that small, locally generated record;
# no model download or outcome work begins before the provider clock is known.
for _ in $(seq 1 120); do
  [ -s "$BUDGET_FILE" ] && break
  sleep 5
done
[ -s "$BUDGET_FILE" ] || {
  echo "FATAL: provider budget metadata did not arrive within 600 seconds"
  exit 2
}

read -r PROVIDER_RATE PROVIDER_START PROVIDER_CAP < <(python3 - "$BUDGET_FILE" "$EXPECTED_COMMIT" <<'PY'
import json, math, sys, time
path, expected_commit = sys.argv[1:]
value = json.load(open(path))
if value.get("schema") != "precision_probe_p02_provider_budget_v1":
    raise SystemExit("FATAL: provider budget schema differs")
if value.get("protocol_id") != "precision-probe-p02":
    raise SystemExit("FATAL: provider budget protocol differs")
if value.get("preregistration_sha256") != "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d":
    raise SystemExit("FATAL: provider budget preregistration binding differs")
if value.get("expected_commit") != expected_commit:
    raise SystemExit("FATAL: provider budget commit differs")
rate = value.get("hourly_cost_usd")
start = value.get("provider_clock_started_epoch")
cap = value.get("provider_wall_cap_seconds")
if (isinstance(rate, bool) or not isinstance(rate, (int, float)) or
        not math.isfinite(rate) or rate <= 0):
    raise SystemExit("FATAL: provider rate is invalid")
if isinstance(start, bool) or not isinstance(start, int) or start <= 0:
    raise SystemExit("FATAL: provider start is invalid")
if isinstance(cap, bool) or not isinstance(cap, int) or not 1 <= cap <= 9000:
    raise SystemExit("FATAL: provider cap is invalid")
expected_cap = min(9000, math.floor(3.90 * 3600 / float(rate)))
absolute_cap = math.floor(4.00 * 3600 / float(rate))
if cap != expected_cap or value.get("scientific_budget_usd") != 3.90:
    raise SystemExit("FATAL: provider scientific cap differs")
if (value.get("absolute_envelope_usd") != 4.00 or
        value.get("absolute_envelope_seconds") != absolute_cap or
        value.get("operational_interrupt_elapsed_seconds") != cap - 210 or
        value.get("forced_pull_elapsed_seconds") != cap - 120 or
        value.get("forced_finalize_elapsed_seconds") != cap - 60 or
        cap * float(rate) / 3600 > 3.90 + 1e-9):
    raise SystemExit("FATAL: provider absolute envelope differs")
if start > int(time.time()) + 60:
    raise SystemExit("FATAL: provider start lies in the future")
print(float(rate), start, cap)
PY
)
echo "PROVIDER_BUDGET rate=$PROVIDER_RATE start_epoch=$PROVIDER_START scientific_cap_seconds=$PROVIDER_CAP scientific_usd=3.90 absolute_usd=4.00"

[ ! -e "$CLONE_TMP" ] || { echo "FATAL: clone path exists: $CLONE_TMP"; exit 3; }
[ ! -e "$REPO" ] || { echo "FATAL: repository path exists: $REPO"; exit 3; }
git clone --quiet --no-checkout https://github.com/jeremyBanks/ValueGraft.git "$CLONE_TMP"
cd "$CLONE_TMP"
git cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null || {
  echo "FATAL: expected commit is absent from origin"
  exit 3
}
git merge-base --is-ancestor "$EXPECTED_COMMIT" origin/trunk || {
  echo "FATAL: expected commit is not an ancestor of origin/trunk"
  exit 3
}
git checkout --quiet -B trunk "$EXPECTED_COMMIT"
[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
  echo "FATAL: exact commit checkout failed"
  exit 3
}
mv "$CLONE_TMP" "$REPO"
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "FATAL: clone is dirty"; exit 3; }
printf '%s  %s\n' "$PREREG_SHA256" "$PREREG" | sha256sum -c -
[ -f scripts/run_precision_probe_p02.py ] || {
  echo "FATAL: p02 runner is absent at the exact commit"
  exit 3
}
[ -f "$PACKAGER" ] || { echo "FATAL: historical packager is absent"; exit 3; }

python3 -m pip install -q --upgrade pip "uv==0.9.18"
uv python install 3.12.11
uv venv --python 3.12.11 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" \
  "torch==2.12.1" \
  "transformers==4.57.6" \
  "accelerate==1.14.0" \
  "bitsandbytes==0.49.2" \
  "huggingface-hub==0.36.2" \
  "safetensors==0.8.0" \
  "tokenizers==0.22.2"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

RUNTIME_ATTESTATION="$OUTPUT_ROOT/precision-probe-p02-runtime_${MODEL_SLUG}_${STAMP}.json"
mkdir -p "$OUTPUT_ROOT"
"$PY" - "$RUNTIME_ATTESTATION" "$EXPECTED_COMMIT" "$MODEL" "$REVISION" \
  "$PREREG_SHA256" <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import torch

output, commit, model, revision, preregistration_sha256 = sys.argv[1:]
expected = {
    "torch": "2.12.1",
    "transformers": "4.57.6",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.49.2",
    "huggingface-hub": "0.36.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}
observed = {key: importlib.metadata.version(key) for key in expected}
if observed != expected:
    raise SystemExit(f"FATAL: dependency drift: {observed}")
if platform.python_version() != "3.12.11":
    raise SystemExit(f"FATAL: Python drift: {platform.python_version()}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("FATAL: exactly one CUDA device is required")
if torch.cuda.get_device_name(0) != "NVIDIA A100 80GB PCIe":
    raise SystemExit(f"FATAL: GPU differs: {torch.cuda.get_device_name(0)}")
if torch.version.cuda != "13.0":
    raise SystemExit(f"FATAL: Torch CUDA differs: {torch.version.cuda}")
line = subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
    "--format=csv,noheader,nounits",
], text=True).strip()
rows = line.splitlines()
if len(rows) != 1:
    raise SystemExit(f"FATAL: expected one nvidia-smi row: {rows}")
name, uuid, driver, memory = [part.strip() for part in rows[0].split(",", 3)]
def version(value):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", value):
        raise SystemExit(f"FATAL: invalid driver: {value}")
    return tuple(map(int, value.split(".")))
if (name != "NVIDIA A100 80GB PCIe" or not uuid.startswith("GPU-") or
        version(driver) < version("580.65.06") or
        not memory.isdigit() or int(memory) < 80000):
    raise SystemExit(f"FATAL: host admission differs: {line}")
document = {
    "schema": "precision_probe_p02_runtime_attestation_v1",
    "protocol_id": "precision-probe-p02",
    "formal_v12_decision_eligible": False,
    "v12_reentry_authorized": False,
    "component_reuse_does_not_inherit_v12_eligibility": True,
    "preregistration_sha256": preregistration_sha256,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_commit": commit,
    "observed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "model": model,
    "revision": revision,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "dependencies": observed,
    "torch_cuda": torch.version.cuda,
    "gpu": line,
}
Path(output).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
print("RUNTIME_ATTESTED", output)
PY

NOW="$(date +%s)"
ELAPSED=$((NOW - PROVIDER_START))
OPERATIONAL_INTERRUPT_ELAPSED=$((PROVIDER_CAP - RUNNER_INTERRUPT_LEAD_SECONDS))
OPERATIONAL_INTERRUPT_EPOCH=$((PROVIDER_START + OPERATIONAL_INTERRUPT_ELAPSED))
REMAINING=$((OPERATIONAL_INTERRUPT_ELAPSED - ELAPSED))
[ "$REMAINING" -ge 60 ] || {
  echo "FATAL: fewer than 60 runner seconds remain after setup (elapsed=$ELAPSED)"
  exit 4
}
command -v timeout >/dev/null || {
  echo "FATAL: GNU timeout is absent; provider budget cannot be bounded"
  exit 4
}

RUNNER_LOG="$OUTPUT_ROOT/precision-probe-p02-runner-log_${MODEL_SLUG}_${STAMP}.log"
RUN_MARKER="$BOOT/precision_probe_p02_runner_${STAMP}.marker"
touch "$RUN_MARKER"
echo "RUN precision-probe-p02 model=$MODEL revision=$REVISION output=$OUTPUT_ROOT elapsed=$ELAPSED allowance=$REMAINING"
set +e
timeout --signal=INT --kill-after=60s "${REMAINING}s" \
  "$PY" -u scripts/run_precision_probe_p02.py \
    --model "$MODEL" \
    --revision "$REVISION" \
    --output-dir "$OUTPUT_ROOT" \
    --hourly-cost-usd "$PROVIDER_RATE" \
    --provider-elapsed-seconds-at-start "$ELAPSED" \
    --provider-wall-cap-seconds "$PROVIDER_CAP" \
    --allow-download \
  2>&1 | tee "$RUNNER_LOG"
RUNNER_EXIT=${PIPESTATUS[0]}
set -e

mapfile -t RUN_DIR_CANDIDATES < <(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 \
  -type d -name 'precision-probe-p02_Qwen3-30B-A3B-Instruct-2507_*' \
  -newer "$RUN_MARKER" -print | sort)
CURRENT_RUN_DIR=""
RUN_DISCOVERY_EXIT=0
if [ "${#RUN_DIR_CANDIDATES[@]}" -eq 1 ]; then
  CURRENT_RUN_DIR="${RUN_DIR_CANDIDATES[0]}"
else
  RUN_DISCOVERY_EXIT=1
  echo "FATAL: runner created ${#RUN_DIR_CANDIDATES[@]} new run directories, expected 1" >&2
fi

# The runner packages every expensive raw execution before the outcome-blind
# extension decision.  Re-verify those packages here with the same frozen
# historical packager.  A source raw remains under .scratch only when packing
# failed; the outer receipt and puller preserve that recovery material too.
PACKAGING_EXIT=0
PACKAGE_COUNT=0
OUTCOME_PACKAGE_COUNT=0
if [ -n "$CURRENT_RUN_DIR" ]; then
  while IFS= read -r manifest; do
    [ -n "$manifest" ] || continue
    PACKAGE_COUNT=$((PACKAGE_COUNT + 1))
    package="$(dirname "$manifest")"
    set +e
    "$PY" "$PACKAGER" verify "$package"
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
      PACKAGING_EXIT=1
      echo "FATAL: lossless package verification failed for $package" >&2
    fi
  done < <(find "$CURRENT_RUN_DIR" -type f \
    -path '*.lossless-package/manifest.json' -print | sort)
  OUTCOME_PACKAGE_COUNT="$(find "$CURRENT_RUN_DIR" -type f \
    -path '*/precision-probe-p02-outcome-*-raw_Qwen3-30B-A3B-Instruct-2507_*.lossless-package/manifest.json' \
    -print | wc -l | tr -d ' ')"
fi
if [ "$RUN_DISCOVERY_EXIT" -ne 0 ]; then
  PACKAGING_EXIT=1
fi
RECOVERY_RAW_COUNT=0
if [ -n "$CURRENT_RUN_DIR" ]; then
  RECOVERY_RAW_COUNT="$(find "$CURRENT_RUN_DIR" -type f \
    -path '*/.scratch/*.json' -print | wc -l | tr -d ' ')"
fi
if [ "$RUNNER_EXIT" -eq 0 ] && { [ "$PACKAGE_COUNT" -lt 4 ] || \
    [ "$OUTCOME_PACKAGE_COUNT" -lt 2 ]; }; then
  PACKAGING_EXIT=1
  echo "FATAL: successful runner produced too few verified technical/outcome packages" >&2
fi
if [ "$RUNNER_EXIT" -eq 0 ] && [ "$RECOVERY_RAW_COUNT" -ne 0 ]; then
  PACKAGING_EXIT=1
  echo "FATAL: successful runner left recovery raw JSON under .scratch" >&2
fi

JOB_LOG="$OUTPUT_ROOT/precision-probe-p02-job-log_${MODEL_SLUG}_${STAMP}.md"
{
  echo "# Precision probe p02 pod log"
  echo
  echo '```text'
  cat "$BOOT/job.log"
  echo '```'
} > "$JOB_LOG"

PROVIDER_RECORD="$OUTPUT_ROOT/precision-probe-p02-provider_${MODEL_SLUG}_${STAMP}.json"
"$PY" - "$BUDGET_FILE" "$PROVIDER_RECORD" "$ELAPSED" "$RUNNER_EXIT" \
  "$PACKAGING_EXIT" "$PACKAGE_COUNT" "$OUTCOME_PACKAGE_COUNT" \
  "$RECOVERY_RAW_COUNT" \
  "$OPERATIONAL_INTERRUPT_ELAPSED" "$OPERATIONAL_INTERRUPT_EPOCH" <<'PY'
from datetime import datetime, timezone
import json, sys, time
source, output = sys.argv[1:3]
value = json.load(open(source))
value.update({
    "job_recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "provider_elapsed_seconds_at_runner_start": int(sys.argv[3]),
    "provider_elapsed_seconds_at_job_record": (
        int(time.time()) - int(value["provider_clock_started_epoch"])),
    "runner_exit": int(sys.argv[4]),
    "packaging_exit": int(sys.argv[5]),
    "lossless_package_count": int(sys.argv[6]),
    "outcome_package_count": int(sys.argv[7]),
    "recovery_raw_count": int(sys.argv[8]),
    "operational_interrupt_elapsed_seconds": int(sys.argv[9]),
    "operational_interrupt_epoch": int(sys.argv[10]),
})
open(output, "w").write(json.dumps(value, indent=2, sort_keys=True) + "\n")
PY

RECEIPT="$OUTPUT_ROOT/precision-probe-p02-receipt_${MODEL_SLUG}_${STAMP}.json"
"$PY" - "$OUTPUT_ROOT" "$RECEIPT" "$EXPECTED_COMMIT" "$RUNNER_EXIT" \
  "$PACKAGING_EXIT" "$PACKAGE_COUNT" "$OUTCOME_PACKAGE_COUNT" \
  "$RECOVERY_RAW_COUNT" \
  "$PROVIDER_RATE" "$PROVIDER_START" \
  "$PROVIDER_CAP" "$OPERATIONAL_INTERRUPT_ELAPSED" \
  "$OPERATIONAL_INTERRUPT_EPOCH" "$CURRENT_RUN_DIR" \
  "$RUNTIME_ATTESTATION" "$RUNNER_LOG" "$JOB_LOG" "$PROVIDER_RECORD" <<'PY'
from datetime import datetime, timezone
import hashlib, json, math, os, re
from pathlib import Path, PurePosixPath
import subprocess, sys, time

root, receipt = map(Path, sys.argv[1:3])
commit = sys.argv[3]
runner_exit, packaging_exit, package_count, outcome_package_count, recovery_raw_count = map(
    int, sys.argv[4:9])
rate = float(sys.argv[9])
provider_start, provider_cap, operational_interrupt_elapsed, operational_interrupt_epoch = map(
    int, sys.argv[10:14])
run_dir_text = sys.argv[14]
support_paths = [Path(value) for value in sys.argv[15:19]]
runtime_metadata = json.loads(support_paths[0].read_bytes())
provider_metadata = json.loads(support_paths[-1].read_bytes())
expected_dependencies = {
    "torch": "2.12.1", "transformers": "4.57.6",
    "accelerate": "1.14.0", "bitsandbytes": "0.49.2",
    "huggingface-hub": "0.36.2", "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
}
if (runtime_metadata.get("schema") !=
        "precision_probe_p02_runtime_attestation_v1" or
        runtime_metadata.get("protocol_id") != "precision-probe-p02" or
        runtime_metadata.get("formal_v12_decision_eligible") is not False or
        runtime_metadata.get("v12_reentry_authorized") is not False or
        runtime_metadata.get(
            "component_reuse_does_not_inherit_v12_eligibility") is not True or
        runtime_metadata.get("expected_commit") != commit or
        runtime_metadata.get("observed_commit") != commit or
        runtime_metadata.get("model") !=
            "Qwen/Qwen3-30B-A3B-Instruct-2507" or
        runtime_metadata.get("revision") !=
            "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" or
        runtime_metadata.get("preregistration_sha256") !=
            "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d" or
        runtime_metadata.get("python") != "3.12.11" or
        runtime_metadata.get("dependencies") != expected_dependencies or
        runtime_metadata.get("torch_cuda") != "13.0"):
    raise SystemExit("runtime attestation binding differs")

def row(path):
    data = path.read_bytes()
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }

artifact_paths = list(support_paths)
if run_dir_text:
    run_dir = Path(run_dir_text)
    if run_dir.parent != root or not run_dir.is_dir():
        raise SystemExit(f"current runner directory is unsafe/absent: {run_dir}")
    artifact_paths.extend(path for path in run_dir.rglob("*") if path.is_file())
if any(not path.is_file() for path in artifact_paths):
    raise SystemExit("one or more support artifacts are absent")
if len(artifact_paths) != len(set(artifact_paths)):
    raise SystemExit("outer artifact inventory contains duplicate paths")
artifacts = [row(path) for path in sorted(artifact_paths)]
completion_pattern = re.compile(
    r"precision-probe-p02-completion_Qwen3-30B-A3B-Instruct-2507_"
    r"[0-9]{8}T[0-9]{6}(?:[0-9]{6})?Z\.json")
runner_completions = [Path(item["path"]) for item in artifacts
                      if completion_pattern.fullmatch(Path(item["path"]).name)]

completion_ok = False
completion_status = None
completion_path = None
if len(runner_completions) == 1:
    completion_path = runner_completions[0]
    completion = json.loads(completion_path.read_bytes())
    completion_status = completion.get("status")
    expected_completion_fields = {
        "schema", "protocol_id", "formal_v12_decision_eligible",
        "v12_reentry_authorized",
        "component_reuse_does_not_inherit_v12_eligibility",
        "semantic_evidence_eligible", "status", "model", "revision",
        "created_at_utc", "scientific_cap_seconds",
        "matched_repeat1_eligible", "repeat2_rider_status",
        "matched_runtime_gate", "run_manifest", "continuation_decisions",
        "artifact_inventory",
        "recovery_raw_files", "provider_elapsed_seconds",
        "estimated_provider_cost_usd",
    }

    def resolve_result_path(value):
        if not isinstance(value, str):
            raise ValueError("artifact path is not a string")
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe runner artifact path: {value}")
        if value.startswith("results/precision_probe_p02/"):
            return Path(value)
        return completion_path.parent / Path(*pure.parts)

    def verify_binding(binding):
        if not isinstance(binding, dict) or set(binding) != {
                "path", "sha256", "size_bytes"}:
            raise ValueError("runner artifact binding differs")
        path = resolve_result_path(binding["path"])
        observed = row(path)
        if (observed["sha256"] != binding["sha256"] or
                observed["size_bytes"] != binding["size_bytes"]):
            raise ValueError(f"runner artifact binding mismatch: {path}")
        return path

    try:
        if set(completion) != expected_completion_fields:
            raise ValueError("runner completion field set differs")
        manifest_path = verify_binding(completion["run_manifest"])
        inventory = completion["artifact_inventory"]
        if not isinstance(inventory, list) or not inventory:
            raise ValueError("runner completion inventory is empty")
        bound_paths = [verify_binding(binding) for binding in inventory]
        if len(bound_paths) != len(set(bound_paths)):
            raise ValueError("runner completion inventory is duplicated")
        actual_run_paths = {
            path for path in completion_path.parent.rglob("*")
            if path.is_file() and path != completion_path
        }
        if set(bound_paths) != actual_run_paths:
            raise ValueError("runner completion inventory is not exhaustive")
        decisions = completion["continuation_decisions"]
        if not isinstance(decisions, dict) or set(decisions) != {
                "initial_schedule", "bf16_technical_admission",
                "bf16_repeat1", "bf16_repeat2"}:
            raise ValueError("continuation-decision set differs")
        decision_paths = [verify_binding(binding)
                          for binding in decisions.values()]
        if len(decision_paths) != len(set(decision_paths)):
            raise ValueError("continuation-decision bindings are duplicated")
        if not set(decision_paths).issubset(bound_paths):
            raise ValueError("continuation decision is absent from inventory")
        manifest = json.loads(manifest_path.read_bytes())
        if (manifest.get("schema") != "precision_probe_p02_run_manifest_v1" or
                manifest.get("protocol_id") != "precision-probe-p02" or
                manifest.get("formal_v12_decision_eligible") is not False or
                manifest.get("v12_reentry_authorized") is not False or
                manifest.get(
                    "component_reuse_does_not_inherit_v12_eligibility") is not True or
                manifest.get("semantic_evidence_eligible") is not False or
                manifest.get("model") !=
                    "Qwen/Qwen3-30B-A3B-Instruct-2507" or
                manifest.get("revision") !=
                    "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"):
            raise ValueError("runner manifest identity/boundary differs")
        regimes = manifest.get("regimes")
        if not isinstance(regimes, dict) or set(regimes) != {"nf4", "bf16"}:
            raise ValueError("runner manifest regime set differs")
        matched_repeat1_count = 0
        for regime_name in ("nf4", "bf16"):
            regime = regimes[regime_name]
            if (not isinstance(regime, dict) or
                    regime.get("status") != "OUTCOMES_FINISHED"):
                raise ValueError(f"{regime_name} regime did not finish outcomes")
            technical = regime.get("technical")
            if (not isinstance(technical, dict) or
                    technical.get("status") != "PASS" or
                    technical.get("regime") != regime_name or
                    technical.get("recovery_raw_path") is not None or
                    not isinstance(technical.get("raw_package"), dict) or
                    technical["raw_package"].get("verification_status") !=
                        "VERIFIED"):
                raise ValueError(f"{regime_name} technical gate did not pass")
            technical_render = verify_binding(technical.get("renders"))
            if technical_render not in bound_paths:
                raise ValueError(f"{regime_name} technical render is absent from inventory")

            outcomes = regime.get("outcomes")
            if not isinstance(outcomes, list):
                raise ValueError(f"{regime_name} outcomes are absent")
            repeat1 = [row for row in outcomes if isinstance(row, dict) and
                       row.get("case_id") == "e01" and
                       row.get("repeat_index") == 1]
            if len(repeat1) != 1:
                raise ValueError(f"{regime_name} matched repeat 1 differs")
            row1 = repeat1[0]
            package = row1.get("raw_package")
            phase_package = row1.get("phase_package")
            if (row1.get("completion_status") != "COMPLETE" or
                    row1.get("recovery_raw_path") is not None or
                    not isinstance(package, dict) or
                    package.get("verification_status") != "VERIFIED" or
                    not isinstance(phase_package, dict) or
                    phase_package.get("verification_status") != "VERIFIED"):
                raise ValueError(f"{regime_name} matched repeat 1 is incomplete")
            derived_paths = [verify_binding(row1.get(key))
                             for key in ("compact", "renders")]
            if not set(derived_paths).issubset(bound_paths):
                raise ValueError(
                    f"{regime_name} derived repeat-1 artifacts are absent from inventory")
            matched_repeat1_count += 1
        elapsed = completion["provider_elapsed_seconds"]
        estimated = completion["estimated_provider_cost_usd"]
        runtime_gate = completion.get("matched_runtime_gate")
        runtime_gate_ok = (
            isinstance(runtime_gate, dict) and set(runtime_gate) == {
                "status", "compared_runtime_fields",
                "compared_subject_binding_fields", "differing_fields",
                "nf4_matched_view_sha256", "bf16_matched_view_sha256"} and
            runtime_gate.get("status") == "PASS" and
            runtime_gate.get("differing_fields") == [] and
            all(isinstance(runtime_gate.get(key), list) and
                bool(runtime_gate[key]) and
                all(isinstance(value, str) and value
                    for value in runtime_gate[key])
                for key in ("compared_runtime_fields",
                            "compared_subject_binding_fields")) and
            all(isinstance(runtime_gate.get(key), str) and
                re.fullmatch(r"[0-9a-f]{64}", runtime_gate[key])
                for key in ("nf4_matched_view_sha256",
                            "bf16_matched_view_sha256")))
        repeat2_status = completion.get("repeat2_rider_status")
        completion_ok = (
            completion.get("schema") == "precision_probe_p02_completion_v1" and
            completion.get("protocol_id") == "precision-probe-p02" and
            completion.get("formal_v12_decision_eligible") is False and
            completion.get("v12_reentry_authorized") is False and
            completion.get(
                "component_reuse_does_not_inherit_v12_eligibility") is True and
            completion.get("semantic_evidence_eligible") is False and
            completion_status == "COMPLETE" and
            isinstance(completion.get("scientific_cap_seconds"), int) and
            not isinstance(completion["scientific_cap_seconds"], bool) and
            completion["scientific_cap_seconds"] == provider_cap and
            completion.get("model") == "Qwen/Qwen3-30B-A3B-Instruct-2507" and
            completion.get("revision") ==
                "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" and
            completion.get("recovery_raw_files") == [] and
            completion.get("matched_repeat1_eligible") is True and
            repeat2_status in {"NOT_AUTHORIZED", "MATCHED_COMPLETE",
                               "NF4_ONLY_TIMING_STOP",
                               "AUTHORIZED_INCOMPLETE"} and
            runtime_gate_ok and
            isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and
            math.isfinite(float(elapsed)) and 0 <= elapsed <= provider_cap and
            isinstance(estimated, (int, float)) and
            not isinstance(estimated, bool) and
            math.isfinite(float(estimated)) and 0 <= estimated <= 3.90 and
            manifest.get("status") == "COMPLETE" and
            matched_repeat1_count == 2)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"RUNNER COMPLETION INVALID: {exc}", file=sys.stderr)
        completion_ok = False

receipt_elapsed = int(time.time()) - provider_start
terminal_pass = (runner_exit == 0 and packaging_exit == 0 and
                 package_count >= 4 and outcome_package_count >= 2 and
                 recovery_raw_count == 0 and
                 completion_ok and receipt_elapsed <= provider_cap)
document = {
    "schema": "precision_probe_p02_pod_receipt_v1",
    "protocol_id": "precision-probe-p02",
    "preregistration_sha256": "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_commit": commit,
    "observed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "revision": "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    "runner_exit": runner_exit,
    "runner_completion_status": completion_status,
    "runner_completion_path": (
        completion_path.as_posix() if completion_path is not None else None),
    "packaging_exit": packaging_exit,
    "lossless_package_count": package_count,
    "outcome_package_count": outcome_package_count,
    "recovery_raw_count": recovery_raw_count,
    "terminal_status": "PASS" if terminal_pass else "ERROR",
    "formal_v12_decision_eligible": False,
    "v12_reentry_authorized": False,
    "component_reuse_does_not_inherit_v12_eligibility": True,
    "provider": {
        "pod_name": provider_metadata.get("pod_name"),
        "provider_pod_id": provider_metadata.get("provider_pod_id"),
        "preregistration_sha256": "dbcece8f189a0574b776149cf597dc1ce8a3b592981440062c749ddb404e221d",
        "hourly_cost_usd": rate,
        "provider_clock_started_epoch": provider_start,
        "provider_wall_cap_seconds": provider_cap,
        "scientific_work_cap_seconds": provider_cap,
        "scientific_budget_usd": 3.90,
        "operational_interrupt_elapsed_seconds": operational_interrupt_elapsed,
        "operational_interrupt_epoch": operational_interrupt_epoch,
        "forced_pull_elapsed_seconds": provider_cap - 120,
        "forced_pull_epoch": provider_start + provider_cap - 120,
        "forced_finalize_elapsed_seconds": provider_cap - 60,
        "forced_finalize_epoch": provider_start + provider_cap - 60,
        "absolute_envelope_seconds": math.floor(4.00 * 3600 / rate),
        "absolute_envelope_epoch": provider_start + math.floor(4.00 * 3600 / rate),
        "absolute_envelope_usd": 4.0,
        "elapsed_seconds_at_receipt": receipt_elapsed,
    },
    "artifacts": artifacts,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
with os.fdopen(descriptor, "w") as handle:
    json.dump(document, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print("P02 RECEIPT", receipt, document["terminal_status"], len(artifacts))
PY
printf '%s\n' "$RECEIPT" > "${POINTER}.tmp"
mv "${POINTER}.tmp" "$POINTER"

TERMINAL_STATUS="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["terminal_status"])' "$RECEIPT")"
echo "PRECISION_PROBE_P02_COMPLETE $(date -Is) terminal_status=$TERMINAL_STATUS receipt=$RECEIPT"
[ "$TERMINAL_STATUS" = "PASS" ] || exit 7
echo "PRECISION_PROBE_P02_PASS"
