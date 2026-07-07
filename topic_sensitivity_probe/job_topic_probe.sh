#!/usr/bin/env bash
set -euo pipefail

cd /workspace/exp

mkdir -p topic_sensitivity_probe/outputs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -f .hf_key ]; then
  export HF_TOKEN="$(tr -d '\r\n' < .hf_key)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
elif [ -f .huggingface_key ]; then
  export HF_TOKEN="$(tr -d '\r\n' < .huggingface_key)"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

python3 -m pip install -U pip
# Do not upgrade torch on RunPod; use the driver-matched image build.
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

python3 -u topic_sensitivity_probe/qwen_topic_probe.py \
  --model Qwen/Qwen3.6-27B \
  --output topic_sensitivity_probe/outputs/qwen36_topic_probe.json \
  --report topic_sensitivity_probe/outputs/qwen36_topic_probe.md \
  --layers all \
  --trajectory-max-tokens 96 \
  "$@" 2>&1 | tee topic_sensitivity_probe/outputs/qwen36_topic_probe.log
