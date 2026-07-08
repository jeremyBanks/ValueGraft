#!/usr/bin/env bash
set -euo pipefail

# Effect-BOUNDING probe (pre-registered, placebo-controlled, single-number).
# Runs on the pod's EXISTING torch -- NO torch upgrade. Purely behavioral metric
# (no J-lens loaded). Cheap: 43 cases x 4 states x K teacher-forced tokens.

# Run from the probe dir wherever it was synced (both /workspace/exp/... and the
# standalone /workspace/jlens_boundary_probe layouts work: the probe defaults
# --data-dir and --output to ../data/synthetic and ../results, one level up).
if [ -d /workspace/exp/jlens_boundary_probe ]; then
  cd /workspace/exp/jlens_boundary_probe
else
  cd /workspace/jlens_boundary_probe
fi
mkdir -p ../results/effect_bound outputs

echo "START effect-bound probe $(date -Is)"
nvidia-smi || true

# Reduce fragmentation for the 27B + KV-snapshot workload. Do NOT upgrade torch.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Verify CUDA is present on the pod's installed torch BEFORE doing any work.
python3 - <<'PY'
import sys
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available on the pod torch", file=sys.stderr)
    sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
# NOTE: no torch upgrade -- rely on the pod's installed torch.
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow

echo "RUN effect-bound probe $(date -Is)"
python3 -u effect_bound_probe.py \
  --model Qwen/Qwen3.6-27B \
  --alpha "${ALPHA:-0.75}" \
  --window-k "${WINDOW_K:-12}" \
  --n-boot "${N_BOOT:-10000}" \
  --output ../results/effect_bound/summary.json

echo "DONE effect-bound probe $(date -Is)"
ls -lh ../results/effect_bound/summary.json
