#!/usr/bin/env bash
# COMPRESSION SWEEP @ bf16 -- does the value-graft effect CONCENTRATE under
# aggressive compaction? ONE A100 80GB, unattended.
#
# QUESTION: the per-layer champion is NULL under the REALISTIC self-gen summary
# (raw_EB CI [-0.041,+0.043]), but the project's positives (judged +12pp, SWE-Gym
# +0.0156) lived under AGGRESSIVE/BRIEF summaries. So: is there a MONOTONIC
# relationship -- does raw_EB (and content-specificity E-placebo) go positive as
# the summary gets more lossy? A "the effect lives in the aggressive-compaction
# regime" result is legitimate + interesting; a "flat null across compression
# levels" is also a real finding. We report the curve with CIs, honestly.
#
# DESIGN -- vary ONLY the summary COMPRESSION LEVEL; hold EVERYTHING else fixed:
#   * model     Qwen/Qwen3-30B-A3B-Instruct-2507, bf16 (SC_LOAD_DTYPE=bfloat16)
#   * summary   DECLARED self-gen (SC_SELFGEN=1) -- the graft needs the model's OWN
#               summarization act (FINDINGS 07-08); SC_NATIVE_RENDER=0 (pre-rendered
#               corpus, matches how the champion was derived).
#   * corpus    HELD-OUT recovery plants c07..c24 (SC_CONV_START=6 SC_CONV_LIMIT=18),
#               disjoint from the tuning VAL set -> clean out-of-sample.
#   * champion  per-layer layers_30b_bf16.json (SC_CHAMPION_CONFIG overridable to
#               swap in a per-head champion if the head-scan produces one that beats
#               it -- see note below).
#   * placebo   FULL battery: shuffle_pos, shuffle_probe, gauss.
#   * metric    raw_EB (PRIMARY) + content_specificity.e_minus_placebo, both with
#               conversation-clustered bootstrap CIs -- per compression level.
#
# COMPRESSION AXIS (SC_SUMMARY_LEVEL, monotonic least->most summary text):
#   ultra      ~1 sentence (<=25w)          MOST aggressive / most lossy
#   brief      3-5 sentences (SUMMARY_REQUEST_BRIEF)   aggressive
#   medium     ~150 words                   moderate
#   realistic  ~300-500 words (SUMMARY_REQUEST)   the CURRENT NULL condition
#   [prod]     ~300-500 words faithful (OpenHands-style) -- optional 5th point
# The harness MEASURES the realized compression ratio (clean-summary tokens /
# full-context tokens) per conv, so "aggressive vs realistic" is a NUMBER; each
# result carries doc.compression.ratio_mean + the manifest records it. Report
# raw_EB & E-placebo AS A FUNCTION of that ratio.
#
# RENDER REUSE: per level, the self-gen summary is generated ONCE and reused across
# the 3 placebo modes (shared --out-dir; champion+placebo live in the SCORE
# fingerprint only). Different LEVELS use different out-dirs AND the summary-request
# hash is in the render fingerprint, so a level never reuses another level's summary.
#
# bf16 loads on the pod's stock torch 2.4.1; only transformers>=4.57,<5 pinned. No
# torch upgrade, no quant libs.
#
# Env (forwarded by launch_pod.sh): SC_HF_MODEL, SC_TUNE_TAG (30b_bf16),
# SC_LOAD_DTYPE (bfloat16), SC_CONV_START/SC_CONV_LIMIT (6/18 = c07..c24),
# SC_CHAMPION_CONFIG (per-layer champion), SC_LEVELS (space-separated level list;
# default "ultra brief medium realistic").
set -uo pipefail   # NOT -e: one level/placebo failing must not lose the rest.
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TAG="${SC_TUNE_TAG:-30b_bf16}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-bfloat16}"
VAL_START="${SC_CONV_START:-6}"     # c07 (0-based offset into sorted c*)
VAL_LIMIT="${SC_CONV_LIMIT:-18}"    # c07..c24 held out from tuning
CFG="${SC_CHAMPION_CONFIG:-data/champion_configs/layers_${TAG}.json}"
LEVELS="${SC_LEVELS:-ultra brief medium realistic}"
MODES="${SC_PLACEBO_MODES:-shuffle_pos shuffle_probe gauss}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORKBASE=results/champion_validate/_work_compsweep
mkdir -p results/champion_validate "$WORKBASE"

echo "START compression-sweep $(date -Is)  model=$MODEL tag=$TAG dtype=$SC_LOAD_DTYPE"
echo "  champion=$CFG  heldout=[$VAL_START:+$VAL_LIMIT] (c07..c24)  levels='$LEVELS'  placebos='$MODES'"
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

python3 -m pip install -U pip >/dev/null
python3 -m pip install -U "transformers>=4.57.0,<5" accelerate safetensors huggingface_hub pandas pyarrow || true
TV=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null)
case "${TV:-x}" in
  4.5[7-9]*|4.[6-9][0-9]*) echo "transformers ${TV} OK (<5)";;
  *) echo "FATAL: transformers=${TV} not in [4.57,5) -- refusing to run"; exit 3;;
esac

[ -f "$CFG" ] || { echo "FATAL: champion config $CFG missing (synced from data/champion_configs?)"; exit 4; }

# =====================================================================
# SWEEP: for each compression LEVEL, render self-gen summary ONCE (shared
# out-dir) and score the champion vs the full placebo battery.
# =====================================================================
for LEVEL in $LEVELS; do
  WORK="$WORKBASE/$LEVEL"; mkdir -p "$WORK"
  for MODE in $MODES; do
    echo "== LEVEL=$LEVEL placebo=$MODE $(date -Is)  cfg=$CFG convs=[$VAL_START:+$VAL_LIMIT]"
    SC_HF_MODEL="$MODEL" SC_NATIVE_RENDER=0 SC_SELFGEN=1 \
    SC_SUMMARY_LEVEL="$LEVEL" \
    SC_CONV_START="$VAL_START" SC_CONV_LIMIT="$VAL_LIMIT" \
    SC_CHAMPION_CONFIG="$CFG" \
      python3 -u src/cross_arch_probe.py \
        --champion-config "$CFG" --placebo "$MODE" \
        --out-dir "$WORK" 2>&1 | tee "compsweep_${TAG}_${LEVEL}_${MODE}.log"
    src_json="$(ls -t "$WORK"/*.json 2>/dev/null | grep -v /manifest | head -1)"
    if [ -n "$src_json" ]; then
      dst="results/champion_validate/compsweep_${TAG}_${LEVEL}_${MODE}_${STAMP}.json"
      cp "$src_json" "$dst" && echo "SAVED $dst"
    else
      echo "WARN: no result json for level=$LEVEL placebo=$MODE"
    fi
  done
done

# =====================================================================
# FINAL SUMMARY: the COMPRESSION-vs-EFFECT curve.
#   x = measured compression ratio (mean clean-summary tokens / full-context tokens)
#   y1 = raw_EB (conv-clustered CI)         -- graft lift over compaction
#   y2 = content_specificity.e_minus_placebo (conv-clustered CI) -- vs placebo
# CI lower bound > 0 => real at that compression level. Read for MONOTONICITY:
# does the effect grow as the ratio shrinks (summary gets more lossy)?
# =====================================================================
echo ""
echo "======================= COMPRESSION-SWEEP CURVE $(date -Is) ======================="
python3 - "$TAG" "$STAMP" "$LEVELS" "$MODES" <<'PY'
import json, glob, sys
tag, stamp, levels, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
prim = modes[0]  # primary placebo for the headline curve (shuffle_pos by default)
def load(level, mode):
    fs = glob.glob(f"results/champion_validate/compsweep_{tag}_{level}_{mode}_{stamp}.json")
    return json.load(open(fs[0])) if fs else None
print(f"{'level':<11}{'ratio':>8}{'sum_tok':>9}{'raw_EB [lo, hi]':>28}{'E-placebo('+prim+') [lo, hi]':>34}{'n':>5}{'conv':>6}")
rows = []
for level in levels:
    d = load(level, prim)
    if not d:
        print(f"{level:<11}{'(missing)':>8}"); continue
    comp = d.get("compression") or {}
    ratio = comp.get("ratio_mean"); stok = comp.get("mean_summary_tokens")
    eb = d.get("raw_EB") or {}
    cs = (d.get("content_specificity") or {}).get("e_minus_placebo") or {}
    lo, hi, mean = cs.get("lo"), cs.get("hi"), cs.get("mean")
    flag = " *CS CI>0*" if (lo is not None and lo > 0) else ""
    ebflag = " *EB CI>0*" if (eb.get("lo") is not None and eb.get("lo") > 0) else ""
    rr = f"{ratio:.4f}" if isinstance(ratio,(int,float)) else "?"
    st = f"{stok:.0f}" if isinstance(stok,(int,float)) else "?"
    print(f"{level:<11}{rr:>8}{st:>9}   {eb.get('mean')} [{eb.get('lo')}, {eb.get('hi')}]{ebflag}"
          f"   {mean} [{lo}, {hi}]{flag}   n={eb.get('n')} c={eb.get('n_conversations')}")
    rows.append((level, ratio, eb.get('mean'), cs.get('mean'), cs.get('lo')))
print("")
# monotonicity read: sort by ratio ascending (most aggressive first) and show E-placebo trend
usable = [r for r in rows if r[1] is not None and r[3] is not None]
usable.sort(key=lambda r: r[1])
if len(usable) >= 2:
    print("CURVE (most-aggressive -> least, by measured ratio):")
    for level, ratio, ebm, csm, cslo in usable:
        sig = " (CS CI>0)" if (cslo is not None and cslo > 0) else ""
        print(f"  ratio={ratio:.4f}  level={level:<10} raw_EB={ebm}  E-placebo={csm}{sig}")
    xs = [r[1] for r in usable]; ys = [r[3] for r in usable]
    # simple sign-of-trend: is E-placebo higher at the most-aggressive (lowest ratio)?
    trend = "CONCENTRATES under aggressive compaction" if ys[0] is not None and ys[-1] is not None and ys[0] > ys[-1] else "does NOT increase toward aggressive compaction"
    print(f"\nTREND (E-placebo at lowest vs highest ratio): {trend}.")
    print("READ: if E-placebo (and raw_EB) rise monotonically as ratio shrinks AND a "
          "CI lower bound clears 0 at the aggressive end, the effect is compaction-"
          "severity-gated. If flat/no-CI-clears-0 everywhere, it is a flat null.")
else:
    print("Not enough levels parsed for a curve -- check per-level logs.")
PY
echo "COMPRESSION_SWEEP_DONE $(date -Is)"
