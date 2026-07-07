#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

echo "START intervention challenge $(date -Is)"
nvidia-smi || true

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub pandas pyarrow
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

echo "RUN sparse challenge probe $(date -Is)"
python3 -u intervention_challenge_probe.py \
  --model Qwen/Qwen3.6-27B \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
  --layers 16,32,48,62 \
  --top-k 10 \
  --cases "${CASES:-all}" \
  --alpha 0.25 \
  --alpha-sweep 0,0.1,0.25,0.5,0.75,1 \
  --tail-messages 0 \
  --output outputs/qwen36_intervention_challenge_probe.json

echo "RUN sparse challenge analysis $(date -Is)"
python3 -u analyze_intervention_batch.py \
  outputs/qwen36_intervention_challenge_probe.json \
  --top-k 10 \
  --output outputs/qwen36_intervention_challenge_summary.json

echo "DONE intervention challenge $(date -Is)"
ls -lh outputs/qwen36_intervention_challenge_probe.json outputs/qwen36_intervention_challenge_summary.json
