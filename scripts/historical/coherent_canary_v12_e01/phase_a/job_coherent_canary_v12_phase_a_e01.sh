#!/usr/bin/env bash
# External orchestration only: run the frozen A7/B7 exact-subject e01 Phase A.
# This file is deliberately ignored and is not part of the scientific apparatus.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo_phasea
VENV=/workspace/v12-phasea-venv
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
EXPECTED_PLATFORM="Linux-6.8.0-100-generic-x86_64-with-glibc2.35"
EXPECTED_GPU_UUID="GPU-0396c7e5-6997-2154-b2cf-90b57c6f05ea"
CASE="data/coherent_canary_v12/revision2/session_d/e01.json"
CASE_SHA="6a2ad7ae0bf094fa5727e76fb710aaeba7bc72082bd93082a5e9a42e09126090"
TECHNICAL_REPORT="results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_20260712T024859Z.json"
TECHNICAL_REPORT_SHA="a8094b4a3355836cfbe0ec0936342a56e271cd992108d7bda61acd8488ee0b48"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CLONE_TMP="/workspace/repo_phasea_${STAMP}.tmp"
HOURLY_COST_USD=2.00

echo "START V12_PHASE_A_E01 $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION"
echo "CASE=$CASE CASE_SHA=$CASE_SHA TECHNICAL_REPORT=$TECHNICAL_REPORT"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader,nounits
OBSERVED_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits)"
[ "$OBSERVED_GPU_UUID" = "$EXPECTED_GPU_UUID" ] || {
  echo "FATAL: GPU UUID differs before setup: $OBSERVED_GPU_UUID"
  exit 20
}

# The Phase-A runner requires the complete technical runtime fingerprint to be
# identical. Reject a different host platform before installing or loading.
OBSERVED_KERNEL="$(uname -r)"
[ "$OBSERVED_KERNEL" = "6.8.0-100-generic" ] || {
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
  echo "FATAL: expected commit is absent from clone"; exit 3;
}
git merge-base --is-ancestor "$EXPECTED_COMMIT" origin/trunk || {
  echo "FATAL: expected commit is not an ancestor of origin/trunk"; exit 3;
}
git checkout --quiet -B trunk "$EXPECTED_COMMIT"
[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
  echo "FATAL: exact commit checkout failed"; exit 3;
}
mv "$CLONE_TMP" "$REPO"
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "FATAL: cloned repository dirty"; exit 3; }
echo "$CASE_SHA  $CASE" | sha256sum -c -
echo "$TECHNICAL_REPORT_SHA  $TECHNICAL_REPORT" | sha256sum -c -

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

"$PY" - "$EXPECTED_COMMIT" "$EXPECTED_PLATFORM" "$EXPECTED_GPU_UUID" <<'PY'
from pathlib import Path
import importlib.metadata
import platform
import sys
import torch
from coherent_canary_loader import verify_frozen_repository

expected_commit, expected_platform, expected_gpu_uuid = sys.argv[1:4]
expected = {
    "torch": "2.12.1", "transformers": "5.0.0",
    "accelerate": "1.14.0", "safetensors": "0.8.0",
    "huggingface-hub": "1.22.0",
}
observed = {name: importlib.metadata.version(name) for name in expected}
print("SETUP python", sys.version)
print("SETUP platform", platform.platform())
print("SETUP dependencies", observed)
print("SETUP torch_cuda", torch.version.cuda,
      "cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
if observed != expected:
    raise SystemExit(f"FATAL: dependency drift: {observed}")
if platform.platform() != expected_platform:
    raise SystemExit(
        f"FATAL: exact technical platform differs: {platform.platform()}")
if torch.__version__ != "2.12.1+cu130" or torch.version.cuda != "13.0":
    raise SystemExit(
        f"FATAL: Torch/CUDA drift: {torch.__version__}/{torch.version.cuda}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("FATAL: e01 Phase A requires exactly one CUDA device")
observed_gpu_uuid = torch.cuda.get_device_properties(0).uuid
if str(observed_gpu_uuid) != expected_gpu_uuid.removeprefix("GPU-"):
    raise SystemExit(
        f"FATAL: Torch GPU UUID differs: {observed_gpu_uuid}")
release = verify_frozen_repository(Path.cwd())
if release["head"] != expected_commit:
    raise SystemExit("FATAL: frozen verifier head differs from launch head")
print("SETUP gpu", torch.cuda.get_device_name(0))
print("V12 FROZEN VERIFIED", release["inventory_sha256"])
PY

MARKER="$BOOT/v12_phase_a_e01_${STAMP}.marker"
touch "$MARKER"
echo "RUN V12_PHASE_A_E01 $(date -Is)"
set +e
"$PY" -u scripts/run_coherent_canary_v12_phase_a.py \
  --subject exact-subject \
  --case "$CASE" \
  --technical-report "$TECHNICAL_REPORT" \
  --allow-download \
  --hourly-cost-usd "$HOURLY_COST_USD"
RUNNER_EXIT=$?
set -e

mapfile -t RAW_CANDIDATES < <(find results/coherent_canary_v12_phase_a \
  -type f -name 'coherent-canary-v12-phase-a-e01_exact-subject_*.json' \
  -newer "$MARKER" -print | sort)
[ "${#RAW_CANDIDATES[@]}" -eq 1 ] || {
  echo "FATAL: Phase-A runner produced ${#RAW_CANDIDATES[@]} new raw artifacts, expected 1"
  exit 5
}
RAW="${RAW_CANDIDATES[0]}"
REPORT="results/coherent_canary_validation/coherent_canary_v12_phase_a_e01_exact-subject_${STAMP}.json"
set +e
"$PY" scripts/validate_coherent_canary_v12_phase_a.py "$RAW" \
  --repo-root "$REPO" --output "$REPORT"
VALIDATOR_EXIT=$?
set -e
[ -f "$REPORT" ] || { echo "FATAL: Phase-A validator wrote no report"; exit 6; }

LOG_MD="results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_${STAMP}_job.md"
{
  echo "# Exact v12 e01 Phase-A pod log"
  echo
  echo '```text'
  cat "$BOOT/job.log"
  echo '```'
} > "$LOG_MD"

RECEIPT="results/coherent_canary_v12_phase_a/coherent-canary-v12-phase-a-e01_exact-subject_${STAMP}_receipt.json"
"$PY" - "$RAW" "$REPORT" "$LOG_MD" "$RECEIPT" \
  "$EXPECTED_COMMIT" "$RUNNER_EXIT" "$VALIDATOR_EXIT" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

raw, report, log, receipt = map(Path, sys.argv[1:5])
expected_commit, runner_exit, validator_exit = sys.argv[5:8]

def row(path):
    data = path.read_bytes()
    return {"path": path.as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}

raw_document = json.loads(raw.read_text())
report_document = json.loads(report.read_text())
document = {
    "schema": "coherent_state_decision_canary_v12_phase_a_pod_receipt_v1",
    "design_id": "coherent-state-decision-canary-v12",
    "case_id": "e01",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_commit": expected_commit,
    "observed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "runner_exit": int(runner_exit),
    "runner_status": raw_document.get("status"),
    "validator_exit": int(validator_exit),
    "validator_status": report_document.get("status"),
    "semantic_release_eligible": report_document.get(
        "semantic_release_eligible"),
    "treatment_scores_present": raw_document.get(
        "treatment_scores_present"),
    "gpu": subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip(),
    "artifacts": {"raw": row(raw), "validation": row(report),
                  "job_log": row(log)},
}
receipt.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
print("RECEIPT", receipt)
PY

printf '%s\n' "$RECEIPT" > "$BOOT/v12_phase_a_current_receipt.txt.tmp"
mv "$BOOT/v12_phase_a_current_receipt.txt.tmp" \
  "$BOOT/v12_phase_a_current_receipt.txt"

read -r RAW_STATUS REPORT_STATUS RELEASE_ELIGIBLE TREATMENT_PRESENT < <(
  "$PY" - "$RAW" "$REPORT" <<'PY'
import json, sys
raw=json.load(open(sys.argv[1])); report=json.load(open(sys.argv[2]))
print(raw.get("status"), report.get("status"),
      str(report.get("semantic_release_eligible") is True).lower(),
      str(raw.get("treatment_scores_present") is True).lower())
PY
)
echo "V12_PHASE_A_E01_COMPLETE $(date -Is) runner_exit=$RUNNER_EXIT validator_exit=$VALIDATOR_EXIT raw_status=$RAW_STATUS report_status=$REPORT_STATUS release_eligible=$RELEASE_ELIGIBLE treatment_present=$TREATMENT_PRESENT raw=$RAW report=$REPORT receipt=$RECEIPT"

if [ "$RUNNER_EXIT" -ne 0 ] || [ "$VALIDATOR_EXIT" -ne 0 ] || \
   [ "$RAW_STATUS" != "PASS" ] || [ "$TREATMENT_PRESENT" != "false" ]; then
  echo "V12_PHASE_A_E01_INVALID runner or structural validation failed"
  exit 7
fi
case "$REPORT_STATUS" in
  PRETREATMENT_PASS)
    [ "$RELEASE_ELIGIBLE" = "true" ] || exit 7
    echo "V12_PHASE_A_E01_PRETREATMENT_PASS"
    ;;
  ESTIMAND_INADEQUATE)
    [ "$RELEASE_ELIGIBLE" = "false" ] || exit 7
    echo "V12_PHASE_A_E01_ESTIMAND_INADEQUATE"
    ;;
  *)
    echo "V12_PHASE_A_E01_INVALID report_status=$REPORT_STATUS"
    exit 7
    ;;
esac
