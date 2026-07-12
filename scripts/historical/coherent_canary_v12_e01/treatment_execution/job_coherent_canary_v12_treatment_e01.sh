#!/usr/bin/env bash
# Ignored external orchestration for the sealed exact-v12 e01 treatment.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo_treatment
VENV=/workspace/v12-treatment-venv
PINNED_COMMIT="cbdfa481fe08de62bf8178d09ca040716d610f21"
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:-$PINNED_COMMIT}"
EXPECTED_PLATFORM="Linux-6.8.0-100-generic-x86_64-with-glibc2.35"
EXPECTED_KERNEL="6.8.0-100-generic"
CASE="data/coherent_canary_v12/revision2/session_d/e01.json"
CASE_SHA="6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090"
PREREG="COHERENT-STATE-DECISION-CANARY-V12-PREREGISTRATION.md"
PREREG_SHA="fc02e86d007369121accbee69471383200fa3f566183e2ed96102e2cf7428b17"
MANIFEST="results/coherent_canary_validation/coherent_canary_revision4_full_manifest_Qwen3-30B-A3B-Instruct-2507_20260711T205951Z.json"
MANIFEST_SHA="738c176fc08075ba905d4c48b70d095a8da0dca649f382e4d7596e4a406affb2"
BLIND_REVIEW="results/coherent_canary_reviews/reviews/revision4_blind_singleton_independent_codex_20260711T210013Z.json"
BLIND_REVIEW_SHA="4b9eace0b4f4754ca0de6fcd6dc59852374ab5fd804f42bfb06d108da1d1fcda"
PAIRED_REVIEW="results/coherent_canary_reviews/reviews/revision4_paired_diversity_independent_codex_20260711T210013Z.json"
PAIRED_REVIEW_SHA="1909224e4dab2265d9b033d7b016c1a9842a24dce9531cf17d240a54287a375b"
TECHNICAL_REPORT="results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json"
TECHNICAL_REPORT_SHA="a8094b4a3355836cfbe0ec0936342a56e271cd992108d7bda61acd8488ee0b48"
PHASE_A_REPORT="results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_20260712T030622Z.json"
PHASE_A_REPORT_SHA="aabac5f0aa64db8f0db49110583be9a17f8453e1a523f1cbc0e0312ec3c15410"
PHASE_A_RAW="results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_20260712T030639073087Z.json"
PHASE_A_RAW_SHA="2cd7f6f190b9f4e9598844e45e5488779b72cd54a1fd51dd5b8a5b24154d672d"
EXPECTED_RUNTIME_SHA="b6158b989b0b467c46589f3ecbe69acba88645d4cd193dd23d925e2e4cd69484"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
PROVIDER_HOURLY_COST_USD=1.39
RUNNER_WALL_CAP_SECONDS=1500
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CLONE_TMP="/workspace/repo_treatment_${STAMP}.tmp"

[ "$EXPECTED_COMMIT" = "$PINNED_COMMIT" ] || {
  echo "FATAL: treatment commit must equal pinned verifier-valid head"
  exit 2
}

echo "START V12_TREATMENT_E01 $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION"
echo "CASE=$CASE CASE_SHA=$CASE_SHA"
echo "TECHNICAL_REPORT=$TECHNICAL_REPORT PHASE_A_REPORT=$PHASE_A_REPORT"
GPU_ADMISSION="$(nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader,nounits)"
echo "$GPU_ADMISSION"
python3 - "$GPU_ADMISSION" <<'PY'
import sys

lines = sys.argv[1].splitlines()
if len(lines) != 1:
    raise SystemExit(f"FATAL: expected one admission GPU row, got {lines}")
name, uuid, driver, memory = [part.strip() for part in lines[0].split(",", 3)]
if name != "NVIDIA A100 80GB PCIe":
    raise SystemExit(f"FATAL: admission GPU differs: {name}")
if not uuid.startswith("GPU-"):
    raise SystemExit(f"FATAL: admission GPU UUID differs: {uuid}")
if driver != "580.159.04":
    raise SystemExit(f"FATAL: admission driver differs from Phase A: {driver}")
if not memory.isdigit() or int(memory) < 80000:
    raise SystemExit(f"FATAL: admission GPU memory differs: {memory}")
print("V12 TREATMENT HOST ADMITTED", name, uuid, driver, memory)
PY

OBSERVED_KERNEL="$(uname -r)"
[ "$OBSERVED_KERNEL" = "$EXPECTED_KERNEL" ] || {
  echo "FATAL: host kernel differs before setup: $OBSERVED_KERNEL"
  exit 20
}

for token_file in "$BOOT/.hf_key" "$BOOT/.huggingface_key"; do
  if [ -f "$token_file" ]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$token_file")"
    break
  fi
done
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF token absent"; exit 2; }

[ ! -e "$CLONE_TMP" ] || { echo "FATAL: clone path already exists"; exit 3; }
[ ! -e "$REPO" ] || { echo "FATAL: repository path already exists"; exit 3; }
git clone --quiet --no-checkout https://github.com/jeremyBanks/ValueGraft.git "$CLONE_TMP"
cd "$CLONE_TMP"
git cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null || {
  echo "FATAL: expected commit is absent from clone"
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
[ -z "$(git status --porcelain)" ] || {
  echo "FATAL: cloned repository is dirty"
  exit 3
}

sha256sum -c <<EOF
$CASE_SHA  $CASE
$PREREG_SHA  $PREREG
$MANIFEST_SHA  $MANIFEST
$BLIND_REVIEW_SHA  $BLIND_REVIEW
$PAIRED_REVIEW_SHA  $PAIRED_REVIEW
$TECHNICAL_REPORT_SHA  $TECHNICAL_REPORT
$PHASE_A_REPORT_SHA  $PHASE_A_REPORT
$PHASE_A_RAW_SHA  $PHASE_A_RAW
EOF

python3 -m pip install -q --upgrade pip "uv==0.9.18"
uv python install 3.12.11
uv venv --python 3.12.11 "$VENV"
PY="$VENV/bin/python"
uv pip install --python "$PY" \
  "torch==2.12.1" \
  "transformers==5.0.0" \
  "accelerate==1.14.0" \
  "safetensors==0.8.0" \
  "huggingface-hub==1.22.0" \
  "sentencepiece==0.2.1"

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

"$PY" - "$EXPECTED_COMMIT" "$EXPECTED_PLATFORM" "$EXPECTED_RUNTIME_SHA" <<'PY'
from pathlib import Path
import importlib.metadata
import platform
import subprocess
import sys
import torch
from coherent_canary_loader import verify_frozen_repository

expected_commit, expected_platform, expected_runtime_sha = sys.argv[1:4]
expected_dependencies = {
    "torch": "2.12.1", "transformers": "5.0.0",
    "accelerate": "1.14.0", "safetensors": "0.8.0",
    "huggingface-hub": "1.22.0",
}
observed_dependencies = {
    name: importlib.metadata.version(name) for name in expected_dependencies
}
gpu_line = subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
    "--format=csv,noheader,nounits",
], text=True).strip()
lines = gpu_line.splitlines()
if len(lines) != 1:
    raise SystemExit(f"FATAL: expected one nvidia-smi row, got {lines}")
name, uuid, driver, memory = [part.strip() for part in lines[0].split(",", 3)]
if name != "NVIDIA A100 80GB PCIe":
    raise SystemExit(f"FATAL: GPU name differs: {name}")
if driver != "580.159.04":
    raise SystemExit(f"FATAL: host driver differs from Phase A: {driver}")
if int(memory) < 80000:
    raise SystemExit(f"FATAL: GPU memory differs: {memory}")
if observed_dependencies != expected_dependencies:
    raise SystemExit(f"FATAL: dependency drift: {observed_dependencies}")
if platform.platform() != expected_platform:
    raise SystemExit(f"FATAL: platform differs: {platform.platform()}")
if platform.python_version() != "3.12.11":
    raise SystemExit(f"FATAL: Python differs: {platform.python_version()}")
if torch.__version__ != "2.12.1+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(f"FATAL: Torch/CUDA drift: {torch.__version__}/{torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("FATAL: exact treatment requires exactly one CUDA device")
if torch.cuda.get_device_name(0) != "NVIDIA A100 80GB PCIe":
    raise SystemExit(f"FATAL: Torch GPU differs: {torch.cuda.get_device_name(0)}")
release = verify_frozen_repository(Path.cwd())
if release["head"] != expected_commit:
    raise SystemExit("FATAL: frozen verifier head differs from pinned head")
print("SETUP python", sys.version)
print("SETUP platform", platform.platform())
print("SETUP dependencies", observed_dependencies)
print("SETUP torch_cuda", torch.version.cuda,
      "cuda_available", torch.cuda.is_available())
print("SETUP gpu", name, uuid, driver, memory)
print("V12 FROZEN VERIFIED", release["inventory_sha256"])
print("EXPECTED RELEASE RUNTIME", expected_runtime_sha)
PY

MARKER="$BOOT/v12_treatment_e01_${STAMP}.marker"
touch "$MARKER"
echo "RUN V12_TREATMENT_E01 $(date -Is)"
command -v timeout >/dev/null || {
  echo "FATAL: GNU timeout is absent; runner wall cap cannot be enforced"
  exit 4
}
set +e
timeout --signal=INT --kill-after=60s "${RUNNER_WALL_CAP_SECONDS}s" \
  "$PY" -u scripts/run_coherent_canary_v12_treatment.py \
  --subject exact-subject \
  --case "$CASE" \
  --technical-report "$TECHNICAL_REPORT" \
  --phase-a-report "$PHASE_A_REPORT" \
  --allow-download \
  --hourly-cost-usd "$PROVIDER_HOURLY_COST_USD"
RUNNER_EXIT=$?
set -e

mapfile -t RAW_CANDIDATES < <(find results/coherent_canary_v12_treatment \
  -type f -name 'coherent-canary-v12-treatment-e01_exact-subject_*.json' \
  -newer "$MARKER" -print | sort)
[ "${#RAW_CANDIDATES[@]}" -eq 1 ] || {
  echo "FATAL: treatment runner produced ${#RAW_CANDIDATES[@]} new raw artifacts, expected 1"
  exit 5
}
RAW="${RAW_CANDIDATES[0]}"
HARVEST="results/coherent_canary_v12_harvest/coherent-canary-v12-harvest-e01-exact-subject-${STAMP}.json"
[ ! -e "$HARVEST" ] || { echo "FATAL: harvest output already exists"; exit 6; }
set +e
"$PY" -u scripts/harvest_coherent_canary_v12.py \
  --treatment "$RAW" --repo "$REPO" --output "$HARVEST"
HARVESTER_EXIT=$?
set -e

JOB_LOG="results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01-exact-subject-${STAMP}_job.md"
{
  echo "# Exact v12 e01 treatment pod log"
  echo
  echo '```text'
  cat "$BOOT/job.log"
  echo '```'
} > "$JOB_LOG"

RECEIPT="results/coherent_canary_v12_treatment/coherent-canary-v12-treatment-e01-exact-subject-${STAMP}_receipt.json"
"$PY" - "$RAW" "$HARVEST" "$JOB_LOG" "$RECEIPT" \
  "$EXPECTED_COMMIT" "$RUNNER_EXIT" "$HARVESTER_EXIT" \
  "$PROVIDER_HOURLY_COST_USD" "$RUNNER_WALL_CAP_SECONDS" \
  "$EXPECTED_RUNTIME_SHA" "$GPU_ADMISSION" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

raw, harvest, job_log, receipt = map(Path, sys.argv[1:5])
expected_commit, runner_exit, harvester_exit = sys.argv[5:8]
provider_rate = float(sys.argv[8])
runner_wall_cap = int(sys.argv[9])
expected_runtime_sha = sys.argv[10]
gpu_admission = sys.argv[11]

def row(path):
    data = path.read_bytes()
    return {"path": path.as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}

def write_exclusive(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

raw_document = json.loads(raw.read_text())
harvest_document = json.loads(harvest.read_text()) if harvest.is_file() else None
treatment = raw_document.get("treatment")
phase_release = raw_document.get("phase_a_release")
runner_status = raw_document.get("status")
raw_row = row(raw)
harvest_pass = (
    isinstance(treatment, dict) and isinstance(harvest_document, dict) and
    harvest_document.get("schema") ==
    "coherent_state_decision_canary_v12_treatment_harvest_v1" and
    harvest_document.get("case_id") == "e01" and
    harvest_document.get("subject") == "exact-subject" and
    harvest_document.get("semantic_evidence_eligible") is True and
    harvest_document.get("apparatus_integration_only") is False and
    harvest_document.get("phase_a_status") == "PRETREATMENT_PASS" and
    harvest_document.get("treatment_artifact", {}).get("sha256") ==
    raw_row["sha256"] and
    harvest_document.get("available_placebo_control_count") ==
    treatment.get("available_placebo_control_count")
)
harvest_status = "PASS" if harvest_pass else "ERROR"
primary_count = treatment.get("primary_arm_count") if isinstance(treatment, dict) else None
placebo_count = treatment.get("placebo_control_count") if isinstance(treatment, dict) else None
available_count = (
    treatment.get("available_placebo_control_count")
    if isinstance(treatment, dict) else None
)
terminal_pass = (
    int(runner_exit) == 0 and runner_status == "PASS" and
    raw_document.get("schema") ==
    "coherent_state_decision_canary_v12_treatment_run_v1" and
    raw_document.get("design_id") == "coherent-state-decision-canary-v12" and
    raw_document.get("case_id") == "e01" and
    raw_document.get("subject") == "exact-subject" and
    raw_document.get("semantic_evidence_eligible") is True and
    raw_document.get("apparatus_integration_only") is False and
    raw_document.get("phase_a_scores_present") is False and
    raw_document.get("runtime_fingerprint", {}).get("fingerprint_sha256") ==
    expected_runtime_sha and
    isinstance(phase_release, dict) and
    phase_release.get("status") == "PRETREATMENT_PASS" and
    phase_release.get("semantic_release_eligible") is True and
    isinstance(treatment, dict) and
    treatment.get("schema") ==
    "coherent_state_decision_canary_v12_treatment_raw_v1" and
    treatment.get("arm_count") == 34 and primary_count == 31 and
    placebo_count == 3 and isinstance(available_count, int) and
    0 <= available_count <= 3 and isinstance(treatment.get("arms"), list) and
    len(treatment["arms"]) == 34 and
    int(harvester_exit) == 0 and harvest_status == "PASS"
)
document = {
    "schema": "coherent_state_decision_canary_v12_treatment_pod_receipt_v1",
    "design_id": "coherent-state-decision-canary-v12",
    "case_id": "e01",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_commit": expected_commit,
    "observed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "runner_exit": int(runner_exit),
    "runner_status": runner_status,
    "harvester_exit": int(harvester_exit),
    "harvest_status": harvest_status,
    "terminal_status": "PASS" if terminal_pass else "ERROR",
    "semantic_evidence_eligible": raw_document.get(
        "semantic_evidence_eligible"),
    "phase_a_release_status": (
        phase_release.get("status") if isinstance(phase_release, dict) else None),
    "phase_a_release_eligible": (
        phase_release.get("semantic_release_eligible")
        if isinstance(phase_release, dict) else None),
    "primary_arm_count": primary_count,
    "placebo_control_count": placebo_count,
    "available_placebo_control_count": available_count,
    "provider_hourly_cost_usd": provider_rate,
    "runner_wall_cap_seconds": runner_wall_cap,
    "gpu": gpu_admission,
    "runtime_fingerprint_sha256": raw_document.get(
        "runtime_fingerprint", {}).get("fingerprint_sha256"),
    "diagnostic_only": True,
    "formal_v12_decision_eligible": False,
    "aggregate_expansion_authorized": False,
    "artifacts": {
        "raw": raw_row,
        "harvest": row(harvest) if harvest.is_file() else None,
        "job_log": row(job_log),
    },
}
write_exclusive(receipt, document)
print("RECEIPT", receipt)
print(json.dumps({
    "terminal_status": document["terminal_status"],
    "runner_status": runner_status,
    "harvest_status": harvest_status,
    "primary_arm_count": primary_count,
    "placebo_control_count": placebo_count,
    "available_placebo_control_count": available_count,
}, sort_keys=True))
PY

printf '%s\n' "$RECEIPT" > "$BOOT/v12_treatment_e01_current_receipt.txt.tmp"
mv "$BOOT/v12_treatment_e01_current_receipt.txt.tmp" \
  "$BOOT/v12_treatment_e01_current_receipt.txt"

TERMINAL_STATUS="$("$PY" - "$RECEIPT" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["terminal_status"])
PY
)"
echo "V12_TREATMENT_E01_COMPLETE $(date -Is) terminal_status=$TERMINAL_STATUS runner_exit=$RUNNER_EXIT harvester_exit=$HARVESTER_EXIT raw=$RAW harvest=$HARVEST receipt=$RECEIPT"
[ "$TERMINAL_STATUS" = "PASS" ] || {
  echo "V12_TREATMENT_E01_INVALID terminal contract did not pass"
  exit 7
}
echo "V12_TREATMENT_E01_PASS"
