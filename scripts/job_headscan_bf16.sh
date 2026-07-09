#!/usr/bin/env bash
# PER-HEAD CHAMPION SCAN @ bf16 on the PRIMARY model -- DERIVE per-head, SELECT,
# then PLACEBO-VALIDATE head / layer / intersection / union side by side.
#
# Question: at bf16 full resolution on Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE), can
# a PER-HEAD value-graft champion (a subset of (layer, kv-head) slots) BEAT the
# PER-LAYER champion (data/champion_configs/layers_30b_bf16.json)? And does an
# intersection/union COMBINATION of the two beat both? Arbiter = the held-out,
# placebo-controlled, conversation-clustered E - placebo CI (content_specificity).
#
# Self-contained (launch_pod uploads THIS as job.sh only). One unattended pod:
#   STAGE 1  derive  : run_tune_hf.py PHASE=head on VAL convs at bf16 -> per
#                      (layer,kv-head) single-slot marginal profile
#                      (results/tune_head_<tag>/<conv>.json). VAL = c01-c06,n01-n04.
#   STAGE 1b select  : src/head_select.py -> heads_<tag>.json + intersection/union
#                      (VAL-only selection; single pre-registered rule).
#   STAGE 2  validate: cross_arch_probe.py placebo-controlled on the HELD-OUT
#                      recovery plants (c07..c24, DISJOINT from VAL), champion
#                      applied to E AND placebo alike, for EACH config:
#                        heads, layers (per-layer champ), intersection, union.
#                      Shared --out-dir => the expensive render is generated ONCE
#                      and reused across every config x placebo mode.
#
# Corpus: PRE-RENDERED data/synthetic + self-gen summary (SC_NATIVE_RENDER=0,
# SC_SELFGEN=1) -- MATCHES how the champions are derived. bf16 loads on the pod's
# stock torch 2.4.1; NO torch upgrade (that was only a 4-bit/compressed-tensors
# issue). Only transformers>=4.57,<5 is pinned.
#
# Env (forwarded by launch_pod.sh): SC_HF_MODEL, SC_TUNE_TAG (default 30b_bf16),
# SC_LOAD_DTYPE (default bfloat16), SC_CONV_START/SC_CONV_LIMIT (validation slice,
# default 6/18 = c07..c24). Selection knobs (job defaults, not forwarded):
# SC_HEAD_SELECT (positive), SC_HEAD_ALPHA (1.0).
set -uo pipefail   # NOT -e: one placebo/config failing must not lose the rest.

cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TAG="${SC_TUNE_TAG:-30b_bf16}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-bfloat16}"
VAL_START="${SC_CONV_START:-6}"     # c07 (0-based offset into sorted c*)
VAL_LIMIT="${SC_CONV_LIMIT:-18}"    # c07..c24 held out from tuning
export SC_HEAD_SELECT="${SC_HEAD_SELECT:-positive}"
export SC_HEAD_ALPHA="${SC_HEAD_ALPHA:-1.0}"
LAYERS_CFG="${SC_LAYERS_CONFIG:-data/champion_configs/layers_${TAG}.json}"
export SC_LAYERS_CONFIG="$LAYERS_CFG"
HEADS_CFG="data/champion_configs/heads_${TAG}.json"
INTER_CFG="data/champion_configs/heads_x_layers_${TAG}_intersection.json"
UNION_CFG="data/champion_configs/heads_x_layers_${TAG}_union.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK=results/champion_validate/_work_headscan
mkdir -p results/champion_validate "$WORK"

echo "START headscan-bf16 $(date -Is)  model=$MODEL tag=$TAG dtype=$SC_LOAD_DTYPE"
echo "  VAL(derive)=c01-c06,n01-n04   HELDOUT(validate)=[$VAL_START:+$VAL_LIMIT] (c07..c24)"
nvidia-smi || true

# ---- HF token (launch_pod renames .huggingface_key -> .hf_key; accept either).
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

# ---- verify CUDA on the pod's stock torch BEFORE any real work (NO upgrade).
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available on the pod torch", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY

# ---- deps: bf16 30B loads on stock torch; only pin transformers<5 (5.x breaks
# the Qwen3Moe load + graft path). No torch upgrade, no quant libs.
python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow || true
TV=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
case "${TV:-x}" in
  4.5[7-9]*|4.[6-9][0-9]*) echo "transformers ${TV} OK (<5)";;
  *) echo "FATAL: transformers=${TV} not in [4.57,5) -- refusing to run"; exit 3;;
esac

# The per-layer champion is required for the combination stage. It ships in the
# repo for bf16; if absent (a re-derive tag), fail loud rather than skip silently.
[ -f "$LAYERS_CFG" ] || { echo "FATAL: per-layer champion $LAYERS_CFG missing (needed for intersection/union)"; exit 4; }

# =====================================================================
# STAGE 1: DERIVE -- per-(layer, kv-head) single-slot profile on VAL at bf16.
# =====================================================================
echo "== STAGE1 per-head profile (PHASE=head) $(date -Is)"
SC_HF_MODEL="$MODEL" SC_TUNE_TAG="$TAG" SC_TUNE_PHASE=head \
  python3 -u src/run_tune_hf.py 2>&1 | tee "tune_head_${TAG}.log"
ls results/tune_head_${TAG}/*.json >/dev/null 2>&1 || { echo "FATAL: no per-head profile produced"; exit 5; }

# ---- STAGE 1b: SELECT + build head_map champion + intersection/union configs.
echo "== STAGE1b select per-head champion + combinations $(date -Is)"
SC_TUNE_TAG="$TAG" python3 -u src/head_select.py 2>&1 | tee "head_select_${TAG}.log"
[ -f "$HEADS_CFG" ] || { echo "FATAL: per-head champion $HEADS_CFG not built"; exit 6; }

# =====================================================================
# STAGE 2: VALIDATE -- placebo-controlled, held-out, EVERY config x placebo mode.
# Shared --out-dir => render generated ONCE (champion cfg is in the SCORE
# fingerprint only) and reused. Each result copied to a UNIQUE self-announcing path.
# =====================================================================
declare -a CFG_NAMES=("heads" "layers" "intersection" "union")
declare -a CFG_PATHS=("$HEADS_CFG" "$LAYERS_CFG" "$INTER_CFG" "$UNION_CFG")
WORKR="$WORK"
for i in "${!CFG_NAMES[@]}"; do
  NAME="${CFG_NAMES[$i]}"; CFG="${CFG_PATHS[$i]}"
  if [ ! -f "$CFG" ]; then echo "SKIP config=$NAME ($CFG absent)"; continue; fi
  for MODE in shuffle_pos shuffle_probe gauss; do
    echo "== STAGE2 config=$NAME placebo=$MODE $(date -Is)  cfg=$CFG convs=[$VAL_START:+$VAL_LIMIT]"
    SC_HF_MODEL="$MODEL" SC_NATIVE_RENDER=0 SC_SELFGEN=1 \
    SC_CONV_START="$VAL_START" SC_CONV_LIMIT="$VAL_LIMIT" \
    SC_CHAMPION_CONFIG="$CFG" \
      python3 -u src/cross_arch_probe.py \
        --champion-config "$CFG" --placebo "$MODE" \
        --out-dir "$WORKR" 2>&1 | tee "headscan_validate_${TAG}_${NAME}_${MODE}.log"
    src_json="$(ls -t "$WORKR"/*.json 2>/dev/null | head -1)"
    if [ -n "$src_json" ]; then
      dst="results/champion_validate/headscan_${TAG}_${NAME}_${MODE}_${STAMP}.json"
      cp "$src_json" "$dst" && echo "SAVED $dst"
    else
      echo "WARN: no result json for config=$NAME placebo=$MODE"
    fi
  done
done

# =====================================================================
# FINAL SUMMARY: per-head / per-layer / intersection / union side by side.
# =====================================================================
echo ""
echo "======================= HEAD-SCAN SUMMARY $(date -Is) ======================="
echo "  DECISIVE = content_specificity.e_minus_placebo (held-out, conv-clustered)."
echo "  CI lower bound > 0  => content-specific champion. Higher E-placebo mean = better."
python3 - "$TAG" "$STAMP" <<'PY'
import json, glob, sys
tag, stamp = sys.argv[1], sys.argv[2]
order = ["heads", "layers", "intersection", "union"]
modes = ["shuffle_pos", "shuffle_probe", "gauss"]
def load(name, mode):
    fs = glob.glob(f"results/champion_validate/headscan_{tag}_{name}_{mode}_{stamp}.json")
    return json.load(open(fs[0])) if fs else None
print(f"{'config':<13}{'placebo':<15}{'E-B mean':>10}{'  E-placebo [lo, hi]':>26}{'  n_pairs':>9}{'  conv':>6}")
best = None
for name in order:
    for mode in modes:
        d = load(name, mode)
        if not d:
            print(f"{name:<13}{mode:<15}{'(missing)':>10}"); continue
        eb = d.get("raw_EB") or {}
        cs = (d.get("content_specificity") or {}).get("e_minus_placebo") or {}
        npairs = (d.get("content_specificity") or {}).get("n_pairs")
        lo, hi, mean = cs.get("lo"), cs.get("hi"), cs.get("mean")
        flag = " *CI>0*" if (lo is not None and lo > 0) else ""
        print(f"{name:<13}{mode:<15}{str(eb.get('mean')):>10}"
              f"   {mean} [{lo}, {hi}]{flag}   n={npairs}  c={eb.get('n_clusters')}")
        if mode == "shuffle_pos" and mean is not None:
            if best is None or mean > best[1]:
                best = (name, mean, lo, hi)
print("")
if best:
    verdict = "content-specific (CI>0)" if (best[2] is not None and best[2] > 0) else "NOT distinguishable from placebo (CI spans 0)"
    print(f"BEST on primary placebo (shuffle_pos): {best[0]}  E-placebo={best[1]} "
          f"CI=[{best[2]}, {best[3]}] -> {verdict}")
    print("READ: compare 'heads' vs 'layers' E-placebo means (+ CIs) to answer "
          "'does per-head beat per-layer'; the higher CI-lower-bound>0 config among "
          "{heads,layers,intersection,union} is the best overall champion.")
else:
    print("NO results parsed -- check the STAGE2 logs.")
PY
echo "HEADSCAN_BF16_DONE $(date -Is)"
