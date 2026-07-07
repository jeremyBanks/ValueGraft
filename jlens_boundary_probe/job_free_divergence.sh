#!/usr/bin/env bash
set -euo pipefail

cd /workspace/jlens_boundary_probe
mkdir -p outputs

echo "START free-divergence probe $(date -Is)"
nvidia-smi || true

# Use the pod's existing torch (do NOT upgrade torch). Reduce fragmentation for
# the 27B + KV-snapshot workload.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ -f /workspace/.huggingface_key ]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '[:space:]' < /workspace/.huggingface_key)"
fi

python3 -m pip install -U pip
# NOTE: no torch upgrade -- rely on the pod's installed torch.
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub pandas pyarrow
python3 -m pip install -U "git+https://github.com/anthropics/jacobian-lens.git"

echo "RUN free-divergence probe $(date -Is)"
python3 -u free_divergence_probe.py \
  --model Qwen/Qwen3.6-27B \
  --lens-repo neuronpedia/jacobian-lens \
  --lens-revision qwen-n1000 \
  --lens-filename qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt \
  --layers "${LAYERS:-8,16,24,32,40,48,56,62}" \
  --top-k 10 \
  --cases "${CASES:-all}" \
  --alpha 0.75 \
  --max-new-tokens "${MAX_NEW_TOKENS:-36}" \
  --output outputs/qwen36_free_divergence.json

echo "DONE free-divergence probe $(date -Is)"
ls -lh outputs/qwen36_free_divergence.json
