#!/usr/bin/env bash
# Pod job: behavioral K/V-policy sweep (src/kv_sweep.py).
# Launched via scripts/launch_pod.sh, which syncs src+data to /workspace/exp
# and runs this as job.sh. Model via SC_HF_MODEL (default 30B MoE).
set -euo pipefail

cd /workspace/exp

# expandable_segments: reduce fragmentation OOM on the MoE 30B under repeated
# full-snapshot grafts (A + B + one E snapshot live at peak per policy).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SC_HF_MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

echo "START kv_sweep $(date -Is)  model=$SC_HF_MODEL"
nvidia-smi || true

# HF token (launch_pod renames .huggingface_key -> .hf_key; accept either).
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

python3 -m pip uninstall -y torchvision 2>/dev/null; python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers==5.0.*" torch accelerate safetensors huggingface_hub

echo "RUN kv_sweep $(date -Is)"
python3 -u src/kv_sweep.py

echo "DONE kv_sweep $(date -Is)"
ls -lh results/kv_sweep/ || true
