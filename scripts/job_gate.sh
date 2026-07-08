#!/usr/bin/env bash
# CONFIDENCE GATE — validate the redesigned apparatus on the KNOWN-POSITIVE anchor
# (Qwen3-30B-A3B) BEFORE any wide spend. This is the validate-before-trusting gate:
#   (a) POSITIVE CONTROL: self-gen graft reproduces ~+0.14 referent on c01-c12
#       (the convs the original result was measured on). If it doesn't, STOP.
#   (b) identity-graft integrity check passes (smoke.identity_ok).
#   (c) placebo (gauss) raw_EB < real raw_EB (structured state, not injected energy).
#   (d) champion-scan OVERHEAD measured -> calibrates the <25% inline-champion budget.
# Self-gen (SC_SELFGEN=1) is the redesign default: a fixed foreign summary suppresses
# the graft (the mechanism needs the model's OWN summary).
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "GATE START $(date -Is)"
nvidia-smi || true

for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done

python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("GATE FATAL: no CUDA", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY

python3 -m pip uninstall -y torchvision 2>/dev/null
python3 -m pip install -U pip >/dev/null 2>&1
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub >/dev/null 2>&1 || true

# force SELF-GEN (redesign default); no fixed summaries
rm -f data/fixed_summaries.json 2>/dev/null
export SC_SELFGEN=1
export SC_HF_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
export SC_CONV_LIMIT="${SC_CONV_LIMIT:-12}"        # c01-c12 = positive-control convs
export SC_GC_ALPHA="${SC_GC_ALPHA:-0.75}"
export SC_PLACEBO="${SC_PLACEBO:-gauss}"           # placebo control in the same run
export SC_CHAMPION_SCAN="${SC_CHAMPION_SCAN:-6}"   # measures overhead_pct for 25% gate
export SC_TRUST_REMOTE=1

echo "== GATE run: $SC_HF_MODEL self-gen, ${SC_CONV_LIMIT} convs, placebo=$SC_PLACEBO, champion=6 $(date -Is)"
timeout 3000s python3 -u src/cross_arch_probe.py 2>&1 | tee cross_arch_GATE.log
rc=${PIPESTATUS[0]}
echo "== run rc=$rc"

python3 - <<'PY'
import json, glob, os
fs=[f for f in sorted(glob.glob("results/cross_arch/*.json")) if not os.path.basename(f).startswith("_")]
if not fs: print("GATE VERDICT: NO OUTPUT — FAIL"); raise SystemExit
d=json.load(open(fs[-1]))
ref=(d.get("by_category_robust",{}) or {}).get("referent",{}) or {}
raw=ref.get("raw_EB"); ci=ref.get("raw_EB_ci")
sm=d.get("smoke",{}) or {}
pb=(d.get("placebo",{}) or {}).get("raw_EB")
ch=d.get("champion_scan",{}) or {}
over=ch.get("overhead_pct")
print("=================== GATE VERDICT ===================")
print(f"status              : {d.get('status')}")
print(f"referent raw_EB     : {raw}  CI {ci}   (target ~ +0.10..+0.16 POSITIVE)")
print(f"identity_ok         : {sm.get('identity_ok')}  (max_diff {sm.get('identity_max_abs_diff')})")
print(f"alpha0_ok           : {sm.get('alpha0_ok')}   graft_changes {sm.get('graft_changes')}")
print(f"placebo raw_EB      : {pb}   (must be < referent raw_EB)")
print(f"champion overhead_% : {over}  (inline champion if < 25)")
print(f"champion all==unif  : {ch.get('champion_all_regions_raw_EB')} vs {ch.get('uniform_raw_EB')}")
ok = (isinstance(raw,(int,float)) and raw>0.05 and sm.get('identity_ok') and sm.get('alpha0_ok')
      and (pb is None or (isinstance(pb,(int,float)) and pb < raw)))
print(f"\nGATE {'PASS ✅ — cleared to go wide' if ok else 'FAIL ❌ — diagnose before wide spend'}")
print("===================================================")
PY
echo "GATE DONE $(date -Is)"
