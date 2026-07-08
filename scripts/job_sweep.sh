#!/usr/bin/env bash
# WIDE cross-architecture value-graft sweep (REDESIGNED) — self-gen summaries,
# attention-geometry-predicts-SIGN. Runs a MODELS list on ONE pod; launch several
# pods with disjoint MODELS subsets for parallel width. Never aborts on one model.
#
# SELF-GEN (SC_SELFGEN=1): each model summarizes with ITSELF (the graft needs the
# model's own write-time summary; a fixed foreign summary suppresses it). Per-model
# pre_graft_gap (A-B) is reported so differing self-summary quality is accounted for.
#
# EVERY model: full corpus (27 convs), champion scan (fingerprint of WHERE the
# graftable signal lives + per-layer value-alignment + region=all sanity + overhead).
# ANCHORS additionally: placebo (gauss) + alpha dose-response (the sign-mechanism core).
#
# Env: MODELS="repo1 repo2" (required, per-pod subset). ANCHORS default set below;
# a model in ANCHORS gets placebo+alpha. MODEL_TIMEOUT (default 3600s).
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SC_SELFGEN=1
export SC_NATIVE_RENDER=1     # per-model native replies (v2.1)
export SC_BATCHED_RENDER=0   # per-token (Fable: batched bf16 divergence risks the sign)
export SC_CONV_LIMIT="${SC_CONV_LIMIT:-12}"     # per-pod override (24 anchors / 12 breadth)
export SC_GC_ALPHA="${SC_GC_ALPHA:-0.75}"
export SC_CHAMPION_SCAN="${SC_CHAMPION_SCAN:-0}"  # OFF for the core sign-map run (cost); champion = cheap follow-up
export SC_TRUST_REMOTE=1
# WITHIN-MODEL QK-NORM ABLATION (default off): SC_ABLATE_QK_NORM=1 disables QK-norm
# after load (clean causal H1 test). Explicit passthrough so it reaches the python
# run; the harness FAILs LOUD if a model has no QK-norm modules to ablate.
export SC_ABLATE_QK_NORM="${SC_ABLATE_QK_NORM:-0}"
# per-token native render is ~750-900s/conv; scale the timeout with the conv count
# (24 convs -> ~6.5h, 12 convs -> ~3.5h) so anchors don't get killed mid-render.
pkill -9 -f cross_arch_probe 2>/dev/null; sleep 3  # no GPU-sharing races on re-run
MODEL_TIMEOUT="${MODEL_TIMEOUT:-$(( ${SC_CONV_LIMIT:-12} * 900 + 1800 ))}"
ANCHORS="${ANCHORS:-Qwen/Qwen3-30B-A3B-Instruct-2507 Qwen/Qwen3-32B Qwen/Qwen2.5-32B-Instruct google/gemma-4-31B-it google/gemma-4-26B-A4B-it}"
# FULL-DEPTH mode: run the complete assessment (placebo + alpha dose-response + champion scan)
# on EVERY model, at whatever SC_CONV_LIMIT is set (use 24 for confirm-capable CIs).
if [ "${SC_FULL_DEPTH:-0}" = "1" ]; then
  export SC_PLACEBO="${SC_PLACEBO:-gauss}"
  export SC_ALPHA_SWEEP="${SC_ALPHA_SWEEP:-1}"
  export SC_CHAMPION_SCAN="${SC_CHAMPION_SCAN:-6}"
  echo "FULL-DEPTH: placebo=$SC_PLACEBO alpha_sweep=$SC_ALPHA_SWEEP champion=$SC_CHAMPION_SCAN conv=$SC_CONV_LIMIT"
fi

echo "WIDE SWEEP START $(date -Is)"; nvidia-smi || true
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  [ -f "$f" ] && { export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; }
done
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
sys.exit(0 if torch.cuda.is_available() else 1)
PY
python3 -m pip uninstall -y torchvision 2>/dev/null
python3 -m pip install -U accelerate safetensors huggingface_hub >/dev/null 2>&1 || true
# FAIL-CLOSED transformers pin: transformers 5.x breaks weight loading ("automatic weight
# conversion" RuntimeError). Force a known-good 4.x, then VERIFY — refuse to run if it did not take.
python3 -m pip install --force-reinstall "transformers==4.57.1" 2>&1 | tail -2
TV=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
echo "transformers pinned -> ${TV:-MISSING}"
case "${TV:-x}" in
  4.5[7-9]*|4.[6-9]*) echo "transformers ${TV} OK (<5)";;
  *) echo "FATAL: transformers=${TV} is not the required 4.57+/<5 — refusing to run (would fail model load)"; exit 3;;
esac
rm -f data/fixed_summaries.json 2>/dev/null   # enforce self-gen

[ -z "${MODELS:-}" ] && { echo "FATAL: MODELS env required"; exit 2; }
echo "PLAN: $MODELS  (anchors get placebo+alpha)"

for M in $MODELS; do
  slug="$(echo "$M" | tr '/ ' '__')"; log="cross_arch_${slug}.log"
  # anchor? -> add placebo + alpha dose-response
  if echo " $ANCHORS " | grep -q " $M "; then
    [ "${SC_FULL_DEPTH:-0}" = "1" ] || { unset SC_PLACEBO; unset SC_ALPHA_SWEEP; }  # core-only unless full-depth
    echo "== ANCHOR $M (self-gen + champion + placebo + alpha) $(date -Is) -> $log"
  else
    [ "${SC_FULL_DEPTH:-0}" = "1" ] || { unset SC_PLACEBO; unset SC_ALPHA_SWEEP; }
    echo "== MODEL  $M (self-gen + champion) $(date -Is) -> $log"
  fi
  # EARLY-SIGNAL PROBE: score a few convs FIRST so a real scored result (sign + sanity)
  # lands in ~40min instead of a 5h black box. Writes the same result json; the full run
  # overwrites it. Catches a broken/out-of-distribution render per architecture fast.
  PROBE="${SC_PROBE_CONVS:-3}"
  if [ "$PROBE" -gt 0 ] && [ "$PROBE" -lt "${SC_CONV_LIMIT:-12}" ]; then
    echo "== PROBE $M ($PROBE convs, EARLY SIGNAL) $(date -Is)"
    SC_HF_MODEL="$M" SC_CONV_LIMIT="$PROBE" timeout 3000s python3 -u src/cross_arch_probe.py 2>&1 | tail -25
  fi
  SC_HF_MODEL="$M" timeout "${MODEL_TIMEOUT}s" python3 -u src/cross_arch_probe.py 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = "124" ]; then
    echo "  TIMEOUT ${MODEL_TIMEOUT}s -- record + move on"
    python3 - "$M" <<'PY'
import json,sys; from pathlib import Path
m=sys.argv[1]; slug=m.replace("/","__").replace(" ","_")
p=Path("results/cross_arch"); p.mkdir(parents=True,exist_ok=True)
json.dump({"model":m,"status":"ERROR","reason":"hard timeout"},open(p/f"{slug}.json","w"),indent=1)
PY
  fi
done

echo "== SWEEP SUMMARY $(date -Is)"
python3 - <<'PY'
import json,glob,os
for f in sorted(glob.glob("results/cross_arch/*.json")):
    if os.path.basename(f).startswith("_"): continue
    d=json.load(open(f))
    ref=(d.get("by_category_robust",{}) or {}).get("referent",{}) or {}
    hp=d.get("model_hparams",{}) or {}
    print(f"  {d.get('architecture', d.get('model','?')):28s} {d.get('status','?'):11s} "
          f"ref_rawEB={ref.get('raw_EB')} ci={ref.get('raw_EB_ci')} "
          f"kv_heads={hp.get('num_key_value_heads')} gqa={hp.get('gqa_ratio')} qk_norm={hp.get('qk_norm')} "
          f"qk_ablated={d.get('qk_norm_ablated')}/{d.get('n_qk_modules_ablated')}")
PY
echo "WIDE SWEEP DONE $(date -Is)"
