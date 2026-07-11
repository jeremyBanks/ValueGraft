#!/usr/bin/env bash
# INDEPENDENT REPLICATION of the SWE-Gym-tuned per-layer champion. One A100 80GB.
#
# THE TEST: the in-domain champion (data/champion_configs/swegym_tuned_20260710T145330Z
# .json -- layers 12-17,30-35 at alpha=1.0) was SELECTED on a subset of the idx>=215
# trajectory pool and cleared baseline held-out THERE (E-champion-B ~ +0.011). The ONE
# thing untested is whether it holds on a trajectory set NEVER involved in its selection.
# The parquet is EXHAUSTED (all 173 budget-passing trajectories used), BUT the ORIGINAL
# 75 (idx 1-213) were only ever scored with the SCALAR / synthetic-champion arms, NEVER
# the SWE-Gym champion, and were entirely disjoint from this champion's selection. So
# re-scoring them with the fixed champion is a clean, in-band, out-of-sample N=75
# replication -- the decisive test of whether +0.011 is real or pool-specific.
#
# Applies the FIXED champion (no re-tuning) + the scalar + baseline + alpha-sweep +
# placebo + the discrete next-action-match metric, in ONE pass over idx 1-213. Brief
# summaries (the regime the effect lives in). UNIQUE stamped dir (loss lesson).
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-bfloat16}"
export SC_SUMMARY="${SC_SUMMARY:-brief}"
export SC_SWE_MIN_IDX="${SC_SWE_MIN_IDX:-0}"      # original pool idx 1-213
export SC_SWE_N="${SC_SWE_N:-75}"                 # the original 75 passers
export SC_SHARD="${SC_SHARD:-0/1}"
export SC_E_ALPHAS="${SC_E_ALPHAS:-0.5,0.75,1.0}"   # also re-test scalar-alpha on this set
export SC_SWE_PLACEBO="${SC_SWE_PLACEBO:-1}"
CHAMP="${SC_CHAMPION_CONFIG:-data/champion_configs/swegym_tuned_20260710T145330Z.json}"
export SC_CHAMPION_CONFIG="$CHAMP"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="confirm_${STAMP}"

echo "START swegym-confirm $(date -Is)  model=$MODEL champion=$CHAMP"
echo "  ORIGINAL-POOL replication: min_idx=$SC_SWE_MIN_IDX n<=$SC_SWE_N summary=$SC_SUMMARY alphas=$SC_E_ALPHAS -> results/swegym_${TAG}_brief"
nvidia-smi || true
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key /workspace/.huggingface_key; do
  if [ -f "$f" ]; then export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; fi
done
python3 - <<'PY' || exit 1
import sys, torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("FATAL: CUDA not available", file=sys.stderr); sys.exit(1)
print("device", torch.cuda.get_device_name(0))
PY
python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow || true
TV=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
case "${TV:-x}" in 4.5[7-9]*|4.[6-9][0-9]*) echo "transformers ${TV} OK";; *) echo "FATAL transformers ${TV} not in [4.57,5)"; exit 3;; esac
[ -f swegym.parquet ] || { echo "FATAL: swegym.parquet missing on pod"; exit 4; }
[ -f "$CHAMP" ] || { echo "FATAL: champion config $CHAMP missing (synced from data/champion_configs?)"; exit 5; }

echo "== run replication $(date -Is)"
SC_HF_MODEL="$MODEL" SC_SWE_TAG="$TAG" python3 -u src/run_swegym_hf.py 2>&1 | tee "swegym_${TAG}.log"
RUNDIR="results/swegym_${TAG}_brief"

echo "== SUMMARY: independent replication on the original pool (idx 1-213) $(date -Is)"
python3 - "$RUNDIR" <<'PY'
import json, glob, random, sys
rundir=sys.argv[1]
def boot(x,seed=0,nb=10000):
    n=len(x)
    if not n: return None
    rng=random.Random(seed); ms=sorted(sum(x[rng.randrange(n)] for _ in range(n))/n for _ in range(nb))
    return dict(n=n,mean=sum(x)/n,lo=ms[int(.025*nb)],hi=ms[int(.975*nb)],pos=sum(1 for v in x if v>0))
def f(b): return "(none)" if not b else f"mean={b['mean']:+.4f} CI[{b['lo']:+.4f},{b['hi']:+.4f}] {b['pos']}/{b['n']} {'CI>0' if b['lo']>0 else 'spans0'}"
R={}
for fn in glob.glob(rundir+"/t*.json"):
    j=json.load(open(fn)); a=j["arms"]; R[j["idx"]]=a
n=len(R); print(f"  n_trajectories={n}")
def col(arm,ref="B"): return [a[arm]["tf_mean"]-a[ref]["tf_mean"] for a in R.values() if arm in a and ref in a]
print("  E-champion - B (THE replication):", f(boot(col("E-champion"))))
print("  E-champion - E-tuned(scalar)   :", f(boot([a["E-champion"]["tf_mean"]-a["E-tuned"]["tf_mean"] for a in R.values() if "E-champion" in a and "E-tuned" in a])))
print("  E-tuned(scalar 0.75) - B       :", f(boot(col("E-tuned"))))
for al in ("0.5","0.75","1"):
    arm=f"E-a{al}"
    if any(arm in a for a in R.values()): print(f"  {arm} - B                       :", f(boot(col(arm))))
# discrete action-match on E-champion vs B
ga=[a for a in R.values() if "E-champion" in a and (a["E-champion"].get("action_match") or {}).get("has_action") is not None]
def am(a,arm): return bool((a.get(arm,{}).get("action_match") or {}).get("match"))
gold=[a for a in R.values() if a.get("gold_action") is not None]
if gold:
    bm=sum(am(a,"B") for a in gold); cm=sum(am(a,"E-champion") for a in gold)
    disc=boot([int(am(a,"E-champion"))-int(am(a,"B")) for a in gold])
    print(f"  [discrete] action-match B={bm}/{len(gold)} E-champion={cm}/{len(gold)}  paired={f(disc)}")
print("\n  READ: E-champion-B CI>0 here => the in-domain champion REPLICATES out-of-pool")
print("        (firms it into a real, modest, in-domain result). CI spans 0 => it was")
print("        specific to the idx>=215 pool it was tuned on (dissolves toward null).")
PY
echo "SWEGYM_CONFIRM_DONE $(date -Is)"
