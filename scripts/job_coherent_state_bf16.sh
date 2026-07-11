#!/usr/bin/env bash
# One-pod Amendments-1-2-3 gapped coherent-summary-state run. The first invocation
# exits intentionally after one scored checkpoint; the second must verify and
# reuse it byte-for-byte.
set -euo pipefail

BOOT=/workspace/exp
REPO=/workspace/repo
EXPECTED_COMMIT="${SC_EXPECTED_COMMIT:?SC_EXPECTED_COMMIT is required}"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
REVISION="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="results/coherent_state/coherent_state_gapped_v3_Qwen3-30B-A3B-Instruct-2507_${STAMP}"
CLONE_TMP="/workspace/repo_${STAMP}.tmp"

echo "START COHERENT_STATE $(date -Is)"
echo "EXPECTED_COMMIT=$EXPECTED_COMMIT MODEL=$MODEL REVISION=$REVISION"
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

echo "PHASE FIRST_INVOCATION_RESUME_PROBE $(date -Is)"
set +e
PYTHONPATH=src python3 -u src/run_coherent_state_hf.py \
  --run-dir "$RUN_DIR" --resume-probe-stop-after-one
FIRST_RC=$?
set -e
[ "$FIRST_RC" = 75 ] || {
  echo "FATAL: first invocation returned $FIRST_RC, expected intentional 75"
  exit 5
}
[ -f "$RUN_DIR/resume_probe.json" ] || {
  echo "FATAL: resume probe marker absent"
  exit 6
}

echo "PHASE SECOND_INVOCATION_RESUME $(date -Is)"
PYTHONPATH=src python3 -u src/run_coherent_state_hf.py --run-dir "$RUN_DIR"

# The harvest validator requires the terminal marker.  Stage a local snapshot of
# the externally captured log with that marker; the watcher later overwrites this
# copy with the complete /workspace/exp/job.log after the real marker is emitted.
cp /workspace/exp/job.log "$RUN_DIR/job.log"
printf '%s\n' "COHERENT_STATE_JOB_DONE" >> "$RUN_DIR/job.log"
python3 scripts/validate_coherent_harvest.py "$RUN_DIR" complete

echo "COHERENT_STATE_JOB_DONE $(date -Is) RUN_DIR=/workspace/repo/$RUN_DIR"
