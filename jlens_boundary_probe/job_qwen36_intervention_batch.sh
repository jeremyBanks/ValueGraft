#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

echo "START intervention batch $(date -Is)"
nvidia-smi || true

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub pandas pyarrow
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

echo "RUN probe $(date -Is)"
python3 -u intervention_batch_probe.py \
  --model Qwen/Qwen3.6-27B \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
  --layers 16,32,48,62 \
  --top-k 8 \
  --cases "${CASES:-all}" \
  --output outputs/qwen36_intervention_batch_probe.json

echo "RUN analysis $(date -Is)"
python3 -u analyze_intervention_batch.py \
  outputs/qwen36_intervention_batch_probe.json \
  --output outputs/qwen36_intervention_batch_summary.json

echo "DONE intervention batch $(date -Is)"
ls -lh outputs/qwen36_intervention_batch_probe.json outputs/qwen36_intervention_batch_summary.json
