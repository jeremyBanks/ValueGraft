#!/usr/bin/env bash
# One-pod Amendments-1-2-3-4 eager technical-only authorization attempt.
# No conversation render, calibration target, A_full, or treatment outcome may run.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/coherent_state/coherent_state_gapped_v4_Qwen3-30B-A3B-Instruct-2507_${STAMP}"
CLONE_TMP="/workspace/repo_${STAMP}.tmp"

echo "START COHERENT_STATE $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION ATTENTION_BACKEND=eager MODE=technical-only"
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
if [ -e "$REPO" ]; then
  echo "FATAL: $REPO already exists before fresh run"
  exit 4
fi
mv "$CLONE_TMP" "$REPO"
cd "$REPO"

python3 -m pip install -q --upgrade pip
python3 -m pip install -q "transformers==5.0.0" "accelerate>=1.14.0" \
  safetensors huggingface_hub sentencepiece
python3 - <<'PY'
import sys, torch, transformers
print("SETUP python", sys.version)
print("SETUP torch", torch.__version__, "transformers", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA unavailable")
print("SETUP gpu", torch.cuda.get_device_name(0))
PY

echo "PHASE EXACT_MODEL_TECHNICAL_ONLY $(date -Is)"
PYTHONPATH=src python3 -u src/run_coherent_state_hf.py \
  --run-dir "$RUN_DIR" --technical-only

# The harvest validator requires the terminal marker.  Stage a local snapshot of
# the externally captured log with that marker; the watcher later overwrites this
# copy with the complete /workspace/exp/job.log after the real marker is emitted.
cp /workspace/exp/job.log "$RUN_DIR/job.log"
printf '%s\n' "COHERENT_STATE_TECHNICAL_DONE" >> "$RUN_DIR/job.log"
python3 scripts/validate_coherent_harvest.py "$RUN_DIR" technical

echo "COHERENT_STATE_TECHNICAL_DONE $(date -Is) RUN_DIR=/workspace/repo/$RUN_DIR"
