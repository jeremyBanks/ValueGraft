#!/usr/bin/env bash
# Pod job: per-layer KEY-graft probe (src/kv_layer_probe.py).
# Does ANY single layer benefit from key-grafting where the coarse uniform
# sweep (job_kv_sweep.sh) found K-graft doesn't help? Launched via
# scripts/launch_pod.sh, which syncs src+data to /workspace/exp and runs this
# as job.sh. Model via SC_HF_MODEL (default 30B MoE).
#
# NO torch upgrade: use the pod's driver-matched torch (upgrading broke CUDA on
# the community pod). This is the ONLY difference in setup vs job_kv_sweep.sh.
set -euo pipefail

cd /workspace/exp

# expandable_segments: reduce fragmentation OOM on the MoE 30B under repeated
# single-layer grafts (A + B + v_snap + one single-layer E snapshot at peak).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SC_HF_MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
# Layer sampling / category knobs (see kv_layer_probe.py header). Defaults:
# every-4th layer (48 -> 12 layers), all three categories.
export SC_LAYER_STRIDE="${SC_LAYER_STRIDE:-4}"
export SC_PROBE_CATS="${SC_PROBE_CATS:-sense,referent,stance}"

echo "START kv_layer_probe $(date -Is)  model=$SC_HF_MODEL  stride=$SC_LAYER_STRIDE  cats=$SC_PROBE_CATS"
nvidia-smi || true

# HF token (launch_pod renames .huggingface_key -> .hf_key; accept either).
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

# Use the pod's existing torch (NO -U torch). Only ensure the support libs.
python3 -m pip uninstall -y torchvision 2>/dev/null || true
python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers==5.0.*" accelerate safetensors huggingface_hub

echo "PLAN kv_layer_probe $(date -Is)"
python3 -u src/kv_layer_probe.py --dry-run || true

echo "RUN kv_layer_probe $(date -Is)"
python3 -u src/kv_layer_probe.py

echo "DONE kv_layer_probe $(date -Is)"
ls -lh results/kv_layer_probe/ || true
