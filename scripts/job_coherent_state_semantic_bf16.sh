#!/usr/bin/env bash
# Separate Amendment-7 semantic process, cryptographically bound to a committed
# technical PASS.  This script is inventoried by the technical attestation.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
TECHNICAL_RESULT_COMMIT="${SC_TECHNICAL_RESULT_COMMIT:?SC_TECHNICAL_RESULT_COMMIT is required}"
TECHNICAL_RUN_DIR="${SC_TECHNICAL_RUN_DIR:?SC_TECHNICAL_RUN_DIR is required}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${SC_SEMANTIC_RUN_DIR:-results/coherent_state/coherent_state_gapped_v7_semantic_Qwen3-30B-A3B-Instruct-2507_${STAMP}}"
CLONE_TMP="/workspace/repo_${STAMP}.tmp"

echo "START COHERENT_STATE_SEMANTIC $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT TECHNICAL_RESULT_COMMIT=$TECHNICAL_RESULT_COMMIT"
echo "TECHNICAL_RUN_DIR=$TECHNICAL_RUN_DIR MODEL=$MODEL REVISION=$REVISION ATTENTION_BACKEND=eager MODE=semantic-authorized"
echo "RUN_DIR=/workspace/repo/$RUN_DIR"
nvidia-smi

for f in "$BOOT/.hf_key" "$BOOT/.huggingface_key" /workspace/.huggingface_key; do
  if [ -f "$f" ]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$f")"
    break
  fi
done
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: HF token absent"; exit 2; }

[ ! -e "$CLONE_TMP" ] || { echo "FATAL: clone path already exists"; exit 3; }
git clone --quiet https://github.com/jeremyBanks/ValueGraft.git "$CLONE_TMP"
cd "$CLONE_TMP"
git checkout --quiet trunk
[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
  echo "FATAL: origin/trunk $(git rev-parse HEAD) != expected $EXPECTED_COMMIT"
  exit 3
}
git merge-base --is-ancestor "$TECHNICAL_RESULT_COMMIT" "$EXPECTED_COMMIT" || {
  echo "FATAL: technical result is not an ancestor of semantic launch"
  exit 3
}
[ -d "$TECHNICAL_RUN_DIR" ] || {
  echo "FATAL: committed technical run directory absent: $TECHNICAL_RUN_DIR"
  exit 3
}
[ -f "${TECHNICAL_RUN_DIR}.harvest_validation.json" ] || {
  echo "FATAL: committed harvest attestation absent"
  exit 3
}
if [ -e "$REPO" ]; then
  echo "FATAL: $REPO already exists before fresh run"
  exit 4
fi
mv "$CLONE_TMP" "$REPO"
cd "$REPO"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q "transformers==5.0.0" "accelerate==1.14.0" \
  "safetensors==0.8.0" "huggingface_hub==1.22.0" "sentencepiece==0.2.1"
python3 - <<'PY'
import torch
print("SETUP torch", torch.__version__, "cuda", torch.version.cuda)
if torch.__version__ != "2.4.1+cu124" or torch.version.cuda != "12.4":
    raise SystemExit(
        f"FATAL: ambient torch/CUDA drifted: {torch.__version__}/{torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA unavailable")
PY

COMMON=(
  --run-dir "$RUN_DIR"
  --semantic-authorization "$TECHNICAL_RUN_DIR"
  --technical-result-commit "$TECHNICAL_RESULT_COMMIT"
)

harvest_semantic_failure() {
  local status="$1"
  if [ -f "$RUN_DIR/terminal_receipt.json" ]; then
    cp /workspace/exp/job.log "$RUN_DIR/job.log"
    printf '%s\n' "COHERENT_STATE_JOB_FAILED" >> "$RUN_DIR/job.log"
    python3 scripts/validate_coherent_harvest.py "$RUN_DIR" failure \
      --output "${RUN_DIR}.harvest_validation.json"
  fi
  echo "COHERENT_STATE_JOB_FAILED $(date -Is) status=$status RUN_DIR=/workspace/repo/$RUN_DIR"
}

echo "PHASE SEMANTIC_AUTHORIZED_RESUME_PROBE $(date -Is)"
set +e
PYTHONPATH=src python3 -u src/run_coherent_state_hf.py \
  "${COMMON[@]}" --resume-probe-stop-after-one
FIRST_STATUS=$?
set -e
[ "$FIRST_STATUS" -eq 75 ] || {
  echo "FATAL: resume-probe process exited $FIRST_STATUS instead of 75"
  harvest_semantic_failure "$FIRST_STATUS"
  [ "$FIRST_STATUS" -ne 0 ] && exit "$FIRST_STATUS"
  exit 1
}

echo "PHASE SEMANTIC_AUTHORIZED_RESUME $(date -Is)"
set +e
PYTHONPATH=src python3 -u src/run_coherent_state_hf.py "${COMMON[@]}"
SECOND_STATUS=$?
set -e
if [ "$SECOND_STATUS" -ne 0 ]; then
  harvest_semantic_failure "$SECOND_STATUS"
  exit "$SECOND_STATUS"
fi

cp /workspace/exp/job.log "$RUN_DIR/job.log"
printf '%s\n' "COHERENT_STATE_JOB_DONE" >> "$RUN_DIR/job.log"
python3 scripts/validate_coherent_harvest.py "$RUN_DIR" complete \
  --output "${RUN_DIR}.harvest_validation.json"
echo "COHERENT_STATE_JOB_DONE $(date -Is) RUN_DIR=/workspace/repo/$RUN_DIR"
