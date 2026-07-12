#!/usr/bin/env bash
# One-pod canary-v12 exact technical gate. No Phase A or treatment is present.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo
VENV=/workspace/v12-venv
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CLONE_TMP="/workspace/repo_${STAMP}.tmp"
HOURLY_COST_USD="${SC_HOURLY_COST_USD:-2.00}"

echo "START V12_EXACT_TECHNICAL $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION"
echo "MODE=exact-technical-only DTYPE=bfloat16 ATTENTION_BACKEND=eager"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total \
  --format=csv,noheader,nounits

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
  echo "FATAL: expected commit is absent from the cloned repository"
  exit 3
}
git merge-base --is-ancestor "$EXPECTED_COMMIT" origin/trunk || {
  echo "FATAL: expected commit is not an ancestor of cloned origin/trunk"
  exit 3
}
git checkout --quiet -B trunk "$EXPECTED_COMMIT"
[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
  echo "FATAL: checked-out commit $(git rev-parse HEAD) != expected $EXPECTED_COMMIT"
  exit 3
}
mv "$CLONE_TMP" "$REPO"
cd "$REPO"
[ -z "$(git status --porcelain)" ] || { echo "FATAL: cloned repository dirty"; exit 3; }

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

"$PY" - <<'PY'
import importlib.metadata
import sys
import torch
expected = {
    "torch": "2.12.1", "transformers": "5.0.0",
    "accelerate": "1.14.0", "safetensors": "0.8.0",
    "huggingface-hub": "1.22.0",
}
observed = {name: importlib.metadata.version(name) for name in expected}
print("SETUP python", sys.version)
print("SETUP dependencies", observed)
print("SETUP torch_cuda", torch.version.cuda,
      "cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
if observed != expected:
    raise SystemExit(f"FATAL: dependency drift: {observed}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("FATAL: exact gate requires exactly one visible CUDA device")
print("SETUP gpu", torch.cuda.get_device_name(0))
PY

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src

echo "RUN V12_EXACT_TECHNICAL $(date -Is)"
RUN_MARKER="$BOOT/v12_exact_technical_${STAMP}.marker"
touch "$RUN_MARKER"
set +e
"$PY" -u scripts/run_coherent_canary_v12_technical.py \
  --subject exact-subject \
  --allow-download \
  --hourly-cost-usd "$HOURLY_COST_USD"
RUNNER_EXIT=$?
set -e

mapfile -t RAW_CANDIDATES < <(find results/coherent_canary_v12_technical \
  -type f -name 'coherent-canary-v12-technical_exact-subject_*.json' \
  -newer "$RUN_MARKER" -print | sort)
[ "${#RAW_CANDIDATES[@]}" -eq 1 ] || {
  echo "FATAL: exact runner produced ${#RAW_CANDIDATES[@]} new raw artifacts, expected 1"
  exit 5
}
RAW="${RAW_CANDIDATES[0]}"
REPORT="results/coherent_canary_validation/coherent_canary_v12_technical_validation_exact-subject_${STAMP}.json"
set +e
"$PY" scripts/validate_coherent_canary_v12_technical.py "$RAW" \
  --repo-root "$REPO" --output "$REPORT"
VALIDATOR_EXIT=$?
set -e
[ -f "$REPORT" ] || { echo "FATAL: independent validator wrote no report"; exit 6; }

LOG_MD="results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_${STAMP}_job.md"
{
  echo "# Exact v12 technical pod log"
  echo
  echo '```text'
  cat "$BOOT/job.log"
  echo '```'
} > "$LOG_MD"

RECEIPT="results/coherent_canary_v12_technical/coherent-canary-v12-technical_exact-subject_${STAMP}_receipt.json"
"$PY" - "$RAW" "$REPORT" "$LOG_MD" "$RECEIPT" \
  "$EXPECTED_COMMIT" "$RUNNER_EXIT" "$VALIDATOR_EXIT" "$HOURLY_COST_USD" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

raw, report, log, receipt = map(Path, sys.argv[1:5])
expected_commit, runner_exit, validator_exit, hourly = sys.argv[5:9]

def row(path):
    data = path.read_bytes()
    return {"path": path.as_posix(), "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}

report_document = json.loads(report.read_text())
gpu = subprocess.check_output([
    "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
    "--format=csv,noheader,nounits"], text=True).strip()
document = {
    "schema": "coherent_state_decision_canary_v12_exact_technical_pod_receipt_v1",
    "design_id": "coherent-state-decision-canary-v12",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "expected_commit": expected_commit,
    "observed_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "runner_exit": int(runner_exit),
    "validator_exit": int(validator_exit),
    "validator_status": report_document.get("status"),
    "semantic_release_eligible": report_document.get("semantic_release_eligible"),
    "hourly_cost_usd_used_for_estimate": float(hourly),
    "gpu": gpu,
    "artifacts": {
        "raw": row(raw), "validation": row(report), "job_log": row(log),
        "identity_checkpoints": [row(path) for path in sorted(
            Path("results/coherent_canary_v12_technical_checkpoints").glob(
                f"{raw.stem}_*.json"))],
    },
}
receipt.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
print("RECEIPT", receipt)
PY
CURRENT_RECEIPT_POINTER="$BOOT/v12_exact_technical_current_receipt.txt"
printf '%s\n' "$RECEIPT" > "${CURRENT_RECEIPT_POINTER}.tmp"
mv "${CURRENT_RECEIPT_POINTER}.tmp" "$CURRENT_RECEIPT_POINTER"

read -r REPORT_STATUS SEMANTIC_ELIGIBLE CHECKPOINT_COUNT < <(
  "$PY" - "$REPORT" "$RECEIPT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
receipt = json.load(open(sys.argv[2]))
print(report["status"],
      str(report.get("semantic_release_eligible") is True).lower(),
      len(receipt["artifacts"]["identity_checkpoints"]))
PY
)
echo "V12_EXACT_TECHNICAL_COMPLETE $(date -Is) runner_exit=$RUNNER_EXIT validator_exit=$VALIDATOR_EXIT report_status=$REPORT_STATUS semantic_release_eligible=$SEMANTIC_ELIGIBLE checkpoint_count=$CHECKPOINT_COUNT raw=$RAW report=$REPORT receipt=$RECEIPT"
if [ "$RUNNER_EXIT" -ne 0 ] || [ "$VALIDATOR_EXIT" -ne 0 ] || \
   [ "$REPORT_STATUS" != "PASS" ] || [ "$SEMANTIC_ELIGIBLE" != "true" ] || \
   [ "$CHECKPOINT_COUNT" -ne 1 ]; then
  echo "V12_EXACT_TECHNICAL_FAILED strict exact gate did not pass"
  exit 7
fi
echo "V12_EXACT_TECHNICAL_PASS"
