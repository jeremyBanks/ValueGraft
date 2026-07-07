#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

echo "START strong-example probe $(date -Is)"
nvidia-smi || true

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub pandas pyarrow
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

echo "RUN strong-example probe $(date -Is)"
python3 -u strong_example_probe.py \
  --model Qwen/Qwen3.6-27B \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
  --layers "${LAYERS:-8,16,24,32,40,48,56,62}" \
  --top-k 10 \
  --cases "${CASES:-all}" \
  --alpha 0.25 \
  --alpha-sweep 0,0.1,0.25,0.5,0.75,1 \
  --tail-messages 0 \
  --output outputs/qwen36_strong_example_probe.json

echo "RUN strong-example analysis $(date -Is)"
python3 -u analyze_intervention_batch.py \
  outputs/qwen36_strong_example_probe.json \
  --top-k 10 \
  --output outputs/qwen36_strong_example_summary.json

echo "DONE strong-example probe $(date -Is)"
ls -lh outputs/qwen36_strong_example_probe.json outputs/qwen36_strong_example_summary.json
