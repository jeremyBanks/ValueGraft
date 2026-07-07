#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "START gap-closure-27b $(date -Is)"
nvidia-smi || true

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub pandas pyarrow

echo "RUN gap-closure-27b $(date -Is)"
python3 -u gap_closure_27b.py \
  --model Qwen/Qwen3.6-27B \
  --alpha "${ALPHA:-0.75}" \
  --output outputs/qwen36_gap_closure.json

echo "DONE gap-closure-27b $(date -Is)"
ls -lh outputs/qwen36_gap_closure.json
