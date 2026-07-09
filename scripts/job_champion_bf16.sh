#!/usr/bin/env bash
# CHAMPION VALIDATION @ bf16 -- PLACEBO-VALIDATE the EXISTING per-layer champion.
#
# The bf16 champion (data/champion_configs/layers_30b_bf16.json) was already
# derived by run_tune_hf on VAL (c01-c06,n01-n04) and is reused here as-is (its
# per-layer marginal ranking is robust to the earlier CI-ratio estimator bug --
# it uses raw score differences). Validation runs on the HELD-OUT recovery plants
# c07..c24 (disjoint from the tuning VAL set) -> clean out-of-sample content-
# specificity. To RE-DERIVE at bf16 instead, run job_champion_4bit.sh with
# SC_HF_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 SC_TUNE_TAG=30b_bf16 SC_LOAD_DTYPE=bfloat16.
#
# Corpus: PRE-RENDERED + self-gen (SC_NATIVE_RENDER=0, SC_SELFGEN=1) -- matches
# how the champion was derived. Decisive quantity: E_champion - placebo_champion
# (paired, conversation-clustered CI) reported as doc.content_specificity.
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TAG="${SC_TUNE_TAG:-30b_bf16}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-bfloat16}"
VAL_START="${SC_CONV_START:-6}"
VAL_LIMIT="${SC_CONV_LIMIT:-18}"
CFG="${SC_CHAMPION_CONFIG:-data/champion_configs/layers_30b_bf16.json}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK=results/champion_validate/_work_bf16
mkdir -p results/champion_validate "$WORK"

echo "START champion-bf16 $(date -Is)  model=$MODEL champion=$CFG"
nvidia-smi || true
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY
python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow || true
[ -f "$CFG" ] || { echo "FATAL: champion config $CFG missing (synced from data/champion_configs?)"; exit 4; }

for MODE in shuffle_pos shuffle_probe gauss; do
  echo "== validate placebo=$MODE $(date -Is)  convs=[$VAL_START:+$VAL_LIMIT]"
  SC_HF_MODEL="$MODEL" SC_NATIVE_RENDER=0 SC_SELFGEN=1 \
  SC_CONV_START="$VAL_START" SC_CONV_LIMIT="$VAL_LIMIT" \
  SC_CHAMPION_CONFIG="$CFG" \
    python3 -u src/cross_arch_probe.py \
      --champion-config "$CFG" --placebo "$MODE" \
      --out-dir "$WORK" 2>&1 | tee "champion_validate_${TAG}_${MODE}.log"
  src_json="$(ls -t "$WORK"/*.json 2>/dev/null | head -1)"
  if [ -n "$src_json" ]; then
    dst="results/champion_validate/champion_${TAG}_${MODE}_${STAMP}.json"
    cp "$src_json" "$dst" && echo "SAVED $dst"
  else
    echo "WARN: no result json for placebo=$MODE"
  fi
done

echo "== SUMMARY $(date -Is)"
python3 - "$TAG" "$STAMP" <<'PY'
import json, glob, sys
tag, stamp = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"results/champion_validate/champion_{tag}_*_{stamp}.json")):
    d = json.load(open(f))
    eb = d.get("raw_EB") or {}
    pl = (d.get("placebo") or {}).get("raw_EB") or {}
    cs = (d.get("content_specificity") or {}).get("e_minus_placebo") or {}
    print(f"  {f.split('/')[-1]}: E-B={eb.get('mean')} ci=[{eb.get('lo')},{eb.get('hi')}] | "
          f"E-placebo={cs.get('mean')} ci=[{cs.get('lo')},{cs.get('hi')}] "
          f"(DECISIVE, CI>0 => content-specific)")
PY
echo "CHAMPION_BF16_DONE $(date -Is)"
