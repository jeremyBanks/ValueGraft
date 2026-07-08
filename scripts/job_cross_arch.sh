#!/usr/bin/env bash
# Generic cross-architecture value-graft gap-closure sweep (src/cross_arch_probe.py).
#
# Tests whether the value-graft gap-closure benefit GENERALIZES beyond Qwen, by
# running the SAME behavioral metric across many ~30B-class architectures (scale
# ~held so ARCHITECTURE is the variable). Uses the pod's EXISTING torch -- NO
# torch upgrade (upgrading has broken CUDA on community pods). Verifies CUDA
# first and bails if absent. Each model runs under a hard `timeout` and NEVER
# crashes the sweep: an unsupported/gated/OOM model writes a status record with a
# reason and we move on.
#
# FIXED SUMMARY (confound fix): the compaction summary TEXT is held IDENTICAL
# across every model, so a model's gap-closure is not confounded by its own
# summary quality. The shared summaries are produced EXTERNALLY (coordinator uses
# Sonnet -- a realistic mid-tier summarizer; a frontier model would write an
# unrealistically good summary and invalidate the test) as
# data/fixed_summaries.json = {conv_id: summary_text}. If that file is absent,
# this script builds one ONCE with a designated summarizer as a fallback.
#
# RUN ORDER: a 3-model PILOT runs first (dense-GQA baseline, a non-Qwen GQA, and
# a sliding-window Gemma) -- those carry most of the evidential value. Set
# PILOT_ONLY=1 to stop after the pilot. Override MODELS=... to run a custom list.
#
# EXCLUDED as a HARD ARCHITECTURAL BLOCK (never run -- NOT gap-closable): Multi-
# head Latent Attention (MLA) models -- Kimi K2.x, native DeepSeek-V3 -- compress
# KV into a shared latent with NO per-head value vectors to graft. The probe has
# nothing to inject; excluded by design, not by failure.
#
# Env: SC_HF_MODEL is set per-model by this script; MODEL_TIMEOUT (default 2700s
# = 45min) hard cap per model; SC_CONV_LIMIT (default 4); SC_GC_ALPHA (0.75).
set -uo pipefail   # NOTE: not -e; one model failing must not abort the sweep.

cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SC_CONV_LIMIT="${SC_CONV_LIMIT:-4}"
export SC_GC_ALPHA="${SC_GC_ALPHA:-0.75}"
export SC_TRUST_REMOTE="${SC_TRUST_REMOTE:-1}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-2700}"
SUMMARIZER="${SC_SUMMARIZER_MODEL:-Qwen/Qwen3.6-27B}"

echo "START cross_arch sweep $(date -Is)"
nvidia-smi || true

# ---- HF token (launch_pod renames .huggingface_key -> .hf_key; accept either).
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

# ---- verify CUDA on the pod's installed torch BEFORE any real work.
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available on the pod torch", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY

# ---- deps (NO torch upgrade). trust_remote_code models (GLM) need recent hub.
python3 -m pip uninstall -y torchvision 2>/dev/null
python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub || true

# ---- LATEST ~30B-class model list (mid-2026). Job skips-with-reason any that
# are gated/unavailable (the probe writes status=UNSUPPORTED/ERROR + reason).
PILOT=(
  "Qwen/Qwen2.5-32B-Instruct"          # dense-GQA baseline (known recipe)
  "mistralai/Mistral-Small-4-Instruct" # different vendor GQA: "not a Qwen thing"
  "google/gemma-4-27b-it"              # SLIDING-WINDOW: "does attn geometry matter"
)
REST=(
  "Qwen/Qwen3.6-35B-A3B"               # latest Qwen MoE GQA
  "Qwen/Qwen3.6-27B"                   # Qwen hybrid -- known-good anchor
  "GLM-4.7-Flash"                      # latest GLM 30B/3B (trust_remote_code)
  "allenai/OLMo-2-0325-32B-Instruct"   # fully-open dense GQA
)

if [ "${MODELS:-}" != "" ]; then
  # shellcheck disable=SC2206
  RUN=($MODELS)
elif [ "${PILOT_ONLY:-0}" = "1" ]; then
  RUN=("${PILOT[@]}")
else
  RUN=("${PILOT[@]}" "${REST[@]}")
fi

echo "PLAN: ${#RUN[@]} models (summarizer=$SUMMARIZER, timeout=${MODEL_TIMEOUT}s each)"

# ---- STEP 1: the shared FIXED summaries (data/fixed_summaries.json) are the
# required input. Prefer the externally-provided (Sonnet-written) file; only
# build one as a fallback if it is missing.
FIXED=data/fixed_summaries.json
if [ -f "$FIXED" ]; then
  echo "== using external fixed summaries: $FIXED ($(python3 -c 'import json,sys;print(len(json.load(open("'"$FIXED"'"))))' 2>/dev/null || echo '?') convs)"
else
  echo "== $FIXED missing -> building fallback summaries with $SUMMARIZER $(date -Is)"
  slug="$(echo "$SUMMARIZER" | tr '/ ' '__')"
  SC_SUMMARIZER_MODEL="$SUMMARIZER" timeout "${MODEL_TIMEOUT}s" \
    python3 -u src/cross_arch_probe.py --make-summaries \
    2>&1 | tee "cross_arch_SUMMARIES_${slug}.log"
fi
if [ ! -f "$FIXED" ]; then
  echo "FATAL: no fixed summaries available; cannot run the sweep" >&2
  exit 4
fi

# ---- STEP 2: run each model under a hard timeout; never abort the sweep.
for M in "${RUN[@]}"; do
  slug="$(echo "$M" | tr '/ ' '__')"
  log="cross_arch_${slug}.log"
  echo "== MODEL $M  $(date -Is)  (timeout ${MODEL_TIMEOUT}s) -> $log"
  SC_HF_MODEL="$M" timeout "${MODEL_TIMEOUT}s" \
    python3 -u src/cross_arch_probe.py 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = "124" ]; then
    echo "  TIMEOUT after ${MODEL_TIMEOUT}s -- recording and moving on"
    python3 - "$M" <<'PY'
import json, sys
from pathlib import Path
m = sys.argv[1]; slug = m.replace("/", "__").replace(" ", "_")
p = Path("results/cross_arch"); p.mkdir(parents=True, exist_ok=True)
json.dump({"model": m, "status": "ERROR", "reason": f"hard timeout"},
          open(p / f"{slug}.json", "w"), indent=1)
PY
  fi
  torch_freed=1  # per-model process exit already frees the GPU
done

echo "== SWEEP SUMMARY $(date -Is)"
python3 - <<'PY'
import json, glob, os
for f in sorted(glob.glob("results/cross_arch/*.json")):
    if os.path.basename(f).startswith("_"): continue
    d = json.load(open(f))
    gc = d.get("gap_closure") or {}
    pg = d.get("pre_graft_gap") or {}
    sm = d.get("smoke") or {}
    print(f"  {d.get('architecture','?'):12s} {d.get('status','?'):11s} "
          f"n={d.get('n_plants',0):>2} "
          f"gc={gc.get('mean')} ci=[{gc.get('lo')},{gc.get('hi')}] "
          f"pre_gap={pg.get('mean')} "
          f"smoke(a0={sm.get('alpha0_ok')},chg={sm.get('graft_changes_output')},"
          f"dir={sm.get('graft_direction_ok')})  {d.get('model')}")
PY
echo "DONE cross_arch sweep $(date -Is)"
