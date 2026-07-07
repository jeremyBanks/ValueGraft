#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

python3 -m pip install -U pip
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

python3 -u boundary_probe.py \
  --model Qwen/Qwen3.6-27B \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
  --layers quarter \
  --top-k 8 \
  --max-new-summary-tokens 96 \
  --output outputs/qwen36_boundary_three_state_probe.json

echo "probe complete: /workspace/jlens_boundary_probe/outputs/qwen36_boundary_three_state_probe.json"
