#!/usr/bin/env bash
# Batched-render timing + reproduction test: does SC_BATCHED_RENDER reproduce +0.10 FAST?
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "BATCHTEST START $(date -Is)"; nvidia-smi || true
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key; do [ -f "$f" ] && { export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; }; done
python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "no CUDA"; exit 1; }
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub >/dev/null 2>&1 || true
export SC_NATIVE_RENDER=1 SC_BATCHED_RENDER=1 SC_SELFGEN=1 SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 SC_CONV_LIMIT=12 SC_CHAMPION_SCAN=0
T0=$(date +%s)
python3 -u src/cross_arch_probe.py 2>&1 | tee batchver.log
echo "== BATCHED render+graft took $(( $(date +%s) - T0 ))s (per-token was ~9000s)"
python3 - <<'PY'
import json,glob,os
fs=[f for f in glob.glob("results/cross_arch/*.json") if not os.path.basename(f).startswith("_")]
d=json.load(open(fs[0])) if fs else {}
r=(d.get("by_category_robust",{}) or {}).get("referent",{}) or {}
print("BATCHTEST referent_ci", r.get("raw_EB_ci"), "(want ~[+0.01,+0.20], midpoint ~+0.10)")
PY
echo "BATCHTEST DONE $(date -Is)"
