#!/usr/bin/env bash
# CHAMPION VALIDATION @ 4-bit -- DERIVE the per-quant champion, then PLACEBO-VALIDATE.
#
# Self-contained (launch_pod only uploads THIS as job.sh; it cannot call other
# scripts/*.sh). Runs, in one unattended pod session:
#   STAGE 1  derive : run_tune_hf.py sweep+layer on VAL convs at 4-bit
#   STAGE 1b build  : rank-based rule -> data/champion_configs/layers_<tag>.json
#                     (the SAME rule that reproduces the bf16 alpha-map:
#                      rank layers by VAL marginal dEB; top ~21% -> a=1.0,
#                      next ~35% positive -> a=0.75, rest -> 0.0)
#   STAGE 2  validate: cross_arch_probe.py placebo-controlled, per placebo mode,
#                     on the HELD-OUT recovery plants (c07..c24), champion config
#                     applied to E AND placebo alike. Decisive: E - placebo CI.
#
# Corpus: PRE-RENDERED data/synthetic + self-gen summary (SC_NATIVE_RENDER=0,
# SC_SELFGEN=1) -- MATCHES how the champion is derived (run_tune_hf uses the same
# pre-rendered convs). Do NOT use the native-render default here: the champion was
# tuned on this corpus.
#
# Env (forwarded by launch_pod.sh): SC_HF_MODEL (the 4-bit repo), SC_TUNE_TAG
# (default 30b_4bit), SC_LOAD_DTYPE (default auto for quantized), SC_CONV_START /
# SC_CONV_LIMIT (validation slice; default 6/18 = c07..c24).
set -uo pipefail   # not -e: a single placebo mode failing must not lose the rest.

cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:?SC_HF_MODEL (the 4-bit repo) is required}"
TAG="${SC_TUNE_TAG:-30b_4bit}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-auto}"   # quantized -> honor quantization_config
VAL_START="${SC_CONV_START:-6}"                 # c07 (0-based offset into sorted c*)
VAL_LIMIT="${SC_CONV_LIMIT:-18}"                # c07..c24 held out from tuning
CFG="data/champion_configs/layers_${TAG}.json"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p results/champion_validate results/champion_validate/_work

echo "START champion-4bit $(date -Is)  model=$MODEL tag=$TAG dtype=$SC_LOAD_DTYPE"
nvidia-smi || true

# ---- HF token (launch_pod renames .huggingface_key -> .hf_key; accept either).
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

# ---- verify CUDA on the pod's installed torch BEFORE any real work (NO upgrade).
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available on the pod torch", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY

# ---- deps. The harness REQUIRES transformers<5 (5.x breaks Qwen3Moe load + the
# graft path). The pitfall we hit: `pip install gptqmodel` prefers transformers>=5
# in its metadata and silently upgrades it, breaking BOTH the harness AND gptqmodel.
# Fix: use a loader that's happy on 4.57 (Intel AutoRound -> auto-round), then FORCE
# transformers back to <5 and VERIFY. This is not a "conflict" — it's pinning.
python3 -m pip install -U pip >/dev/null
# ROOT-CAUSE GUARD: quant libs (compressed-tensors, autoawq, ...) drag in a newer
# torch + CUDA-13 wheels, clobbering the pod's matched torch build and breaking the
# whole CUDA stack (model then never reaches the GPU). Freeze torch to whatever the
# pod shipped, and cap transformers<5, via a pip CONSTRAINT that every install below
# must respect. This is the real fix for the "package conflict" — pin, don't pray.
TORCH_LOCK=$(python3 -c "import torch;print('torch=='+torch.__version__)" 2>/dev/null)
{ echo "$TORCH_LOCK"; echo "transformers<5"; } > /tmp/pip-constraints.txt
export PIP_CONSTRAINT=/tmp/pip-constraints.txt
echo "PIP_CONSTRAINT locking ${TORCH_LOCK} + transformers<5 for all installs"
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow || true
# quant loader for the chosen repo (default: Intel int4-AutoRound). Alternatives if the
# repo is a different quant: `autoawq` (AWQ) or `auto-gptq optimum` (GPTQ, transformers-4.x path).
python3 -m pip install -U "${SC_QUANT_PKG:-auto-round}" || true
# HARD re-pin: whatever the loader dragged in, transformers MUST be <5. Force + verify.
python3 -m pip install -U --force-reinstall "transformers>=4.57.0,<5" 2>&1 | tail -1
TV=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
case "${TV:-x}" in
  4.5[7-9]*|4.[6-9][0-9]*) echo "transformers ${TV} OK (<5), quant loader ${SC_QUANT_PKG:-auto-round}";;
  *) echo "FATAL: transformers=${TV} is not <5 after loader install — refusing to run"; exit 3;;
esac

# =====================================================================
# STAGE 1: DERIVE -- per-layer profile on VAL convs at this quant.
# =====================================================================
echo "== STAGE1 tune sweep $(date -Is)"
SC_HF_MODEL="$MODEL" SC_TUNE_TAG="$TAG" SC_TUNE_PHASE=sweep \
  python3 -u src/run_tune_hf.py 2>&1 | tee "tune_sweep_${TAG}.log"
echo "== STAGE1 tune layer profile $(date -Is)"
SC_HF_MODEL="$MODEL" SC_TUNE_TAG="$TAG" SC_TUNE_PHASE=layer \
  python3 -u src/run_tune_hf.py 2>&1 | tee "tune_layer_${TAG}.log"

# ---- STAGE 1b: BUILD the champion alpha-map from the layer profile (rank rule).
echo "== STAGE1b build champion config -> $CFG $(date -Is)"
SC_TUNE_TAG="$TAG" python3 - "$TAG" <<'PY'
import json, glob, os, sys
from collections import defaultdict
tag = sys.argv[1]
F_HIGH = float(os.environ.get("SC_FRAC_HIGH", "0.21"))   # top share -> alpha=1.0
F_MID  = float(os.environ.get("SC_FRAC_MID",  "0.35"))   # next share -> alpha=0.75
files = glob.glob(f"results/tune_layer_{tag}/*.json")
assert files, f"no layer-profile results in results/tune_layer_{tag}/"
marg = defaultdict(list)
for f in files:
    d = json.load(open(f)); B = d["B"]
    for k, v in d.items():
        if k.startswith("L") and k[1:].isdigit():
            marg[int(k[1:])].append(v - B)
prof = {l: sum(vs) / len(vs) for l, vs in marg.items()}
n = len(prof)
ranked = sorted(prof, key=lambda l: prof[l], reverse=True)
hi = round(F_HIGH * n); mid = round((F_HIGH + F_MID) * n)
amap = {}
for i, l in enumerate(ranked):
    if prof[l] <= 0.0:            a = 0.0     # non-positive marginal: never graft
    elif i < hi:                  a = 1.0
    elif i < mid:                 a = 0.75
    else:                         a = 0.0
    if a > 0.0:
        amap[str(l)] = a
os.makedirs("data/champion_configs", exist_ok=True)
out = f"data/champion_configs/layers_{tag}.json"
json.dump({"label": f"layers_{tag}",
           "note": (f"per-layer champion alpha-map DERIVED at {tag} from the "
                    f"VAL layer profile (rank rule: top {F_HIGH:.0%}->1.0, next "
                    f"{F_MID:.0%} positive->0.75, rest->0.0). Reproduces the bf16 "
                    "derivation on the bf16 profile."),
           "alpha_map": amap},
          open(out, "w"), indent=1)
n1 = sum(1 for v in amap.values() if v == 1.0)
n07 = sum(1 for v in amap.values() if v == 0.75)
print(f"WROTE {out}: {n1} layers @1.0, {n07} @0.75, {n - len(amap)} @0.0 (n_layers={n})")
PY
[ -f "$CFG" ] || { echo "FATAL: champion config $CFG not built"; exit 4; }

# =====================================================================
# STAGE 2: VALIDATE -- placebo-controlled, per placebo mode, held-out convs.
# Shared --out-dir => the expensive render is generated ONCE and REUSED across
# placebo modes (render fingerprint excludes placebo/champion). Each mode's
# result json is copied to a UNIQUE path (never a shared summary.json).
# =====================================================================
WORK=results/champion_validate/_work
for MODE in shuffle_pos shuffle_probe gauss; do
  echo "== STAGE2 validate placebo=$MODE $(date -Is)  champion=$CFG convs=[$VAL_START:+$VAL_LIMIT]"
  SC_HF_MODEL="$MODEL" SC_NATIVE_RENDER=0 SC_SELFGEN=1 \
  SC_CONV_START="$VAL_START" SC_CONV_LIMIT="$VAL_LIMIT" \
  SC_CHAMPION_CONFIG="$CFG" \
    python3 -u src/cross_arch_probe.py \
      --champion-config "$CFG" --placebo "$MODE" \
      --out-dir "$WORK" 2>&1 | tee "champion_validate_${TAG}_${MODE}.log"
  # copy the per-model result json out to a unique, self-announcing path.
  src_json="$(ls -t "$WORK"/*.json 2>/dev/null | head -1)"
  if [ -n "$src_json" ]; then
    dst="results/champion_validate/champion_${TAG}_${MODE}_${STAMP}.json"
    cp "$src_json" "$dst" && echo "SAVED $dst"
  else
    echo "WARN: no result json for placebo=$MODE"
  fi
done

echo "== SUMMARY (E-B, placebo, E-placebo per mode) $(date -Is)"
python3 - "$TAG" "$STAMP" <<'PY'
import json, glob, sys
tag, stamp = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f"results/champion_validate/champion_{tag}_*_{stamp}.json")):
    d = json.load(open(f))
    eb = (d.get("raw_EB") or {})
    pl = (d.get("placebo") or {}).get("raw_EB") or {}
    cs = (d.get("content_specificity") or {}).get("e_minus_placebo") or {}
    print(f"  {f.split('/')[-1]}")
    print(f"    E-B      mean={eb.get('mean')} ci=[{eb.get('lo')},{eb.get('hi')}] "
          f"n={eb.get('n')} conv={eb.get('n_clusters')}")
    print(f"    placebo  mean={pl.get('mean')} ci=[{pl.get('lo')},{pl.get('hi')}]")
    print(f"    E-placebo mean={cs.get('mean')} ci=[{cs.get('lo')},{cs.get('hi')}] "
          f"n_pairs={(d.get('content_specificity') or {}).get('n_pairs')}  "
          f"<-- DECISIVE (CI>0 => content-specific champion)")
PY
echo "CHAMPION_4BIT_DONE $(date -Is)"
