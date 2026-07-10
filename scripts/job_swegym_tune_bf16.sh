#!/usr/bin/env bash
# IN-DOMAIN GRAFT OPTIMIZATION on brief-SWE-Gym -- the ONE regime the value graft
# is net-positive. One A100 80GB, unattended, bf16 stock torch.
#
# THE GAP: the graft's only net-positive cell is brief-SWE-Gym (scalar alpha=0.75
# -> next-action recovery +0.013 [+0.002,+0.026], clears 0). Every champion we
# built was tuned on the SYNTHETIC CONVERSATION corpus (the null regime) and
# transferred out-of-domain adding nothing. We have NEVER swept alpha or profiled
# per-layer ON CODING trajectories. This optimizes the graft in its home regime.
#
# STAGES (SC_STAGE):
#   sweep  (default, cheapest, do FIRST): ONE harness pass over the trajectories
#          scoring the alpha-sweep {0.25,0.5,0.75,1.0,1.5} + a position-shuffle
#          PLACEBO per alpha (content-specificity), all under BRIEF summaries.
#          Analysis: alpha curve, held-out (tune->eval) alpha selection, E-P.
#   full   : the above PLUS per-region layer profiling in the same pass, then a
#          CPU champion build (union of positive TUNE-split regions), then a SECOND
#          harness pass evaluating that SWE-Gym-tuned champion HELD-OUT on the eval
#          split (E-champion vs scalar E-tuned vs B), then final analysis.
#
# HELD-OUT: each trajectory carries a deterministic tune|eval split; alpha is
# selected and the layer champion built on TUNE, reported on the DISJOINT EVAL
# split. With ~75 usable trajectories each split is thin -- the analysis prints the
# overfitting caveat. Metric = teacher-forced next-action logprob (PROXY, not
# resolve rate); brief summary; born-annotated manifest per run.
#
# LOSS LESSON: every run writes to a UNIQUE, STAMPED directory. This job NEVER
# writes to results/swegym_30b_bf16_{brief,prod} and never reuses a prior run's
# filenames -- the prod-champion data was lost exactly that way.
set -uo pipefail   # NOT -e: one stage failing must not lose the earlier ones.
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
export SC_LOAD_DTYPE="${SC_LOAD_DTYPE:-bfloat16}"
export SC_SUMMARY="${SC_SUMMARY:-brief}"          # the regime the effect lives in
# DISJOINT NEW N to resolve the +0.013 positive's SIGN: the existing two 75-traj
# runs consumed find_cut-passing indices in [1,213]; MIN_IDX=214 scores a
# GUARANTEED-disjoint set (independent N, not a re-measurement). The parquet has
# 491 trajectories but only ~35% pass the find_cut budget filter, so the disjoint
# tail yields ~90-100 usable (SC_SWE_N is a ceiling; the run takes what passes).
export SC_SWE_MIN_IDX="${SC_SWE_MIN_IDX:-214}"
export SC_SWE_N="${SC_SWE_N:-150}"
export SC_SHARD="${SC_SHARD:-0/1}"
export SC_E_ALPHAS="${SC_E_ALPHAS:-0.25,0.5,0.75,1.0,1.5}"
export SC_SWE_PLACEBO="${SC_SWE_PLACEBO:-1}"
STAGE="${SC_STAGE:-sweep}"
REGIONS="${SC_PROFILE_REGIONS:-8}"                # used only in full mode
export SC_PROFILE_ALPHA="${SC_PROFILE_ALPHA:-1.0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG1="tune_${STAMP}"                              # -> results/swegym_tune_<stamp>_brief
CHAMP="data/champion_configs/swegym_tuned_${STAMP}.json"
mkdir -p data/champion_configs

echo "START swegym-tune $(date -Is)  model=$MODEL stage=$STAGE summary=$SC_SUMMARY"
echo "  DISJOINT min_idx=$SC_SWE_MIN_IDX n<=$SC_SWE_N alphas=$SC_E_ALPHAS placebo=$SC_SWE_PLACEBO regions(full)=$REGIONS stamp=$STAMP"
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
[ -f swegym.parquet ] || { echo "FATAL: swegym.parquet missing on pod (stage it locally so launch_pod rsyncs it)"; exit 4; }

# =====================================================================
# STAGE 1: one pass -- alpha-sweep + placebo (+ region profiling in full mode).
# =====================================================================
if [ "$STAGE" = "full" ]; then export SC_PROFILE_REGIONS="$REGIONS"; else unset SC_PROFILE_REGIONS; fi
echo "== STAGE1 sweep+placebo${SC_PROFILE_REGIONS:++profile} $(date -Is) -> results/swegym_${TAG1}_brief"
SC_HF_MODEL="$MODEL" SC_SWE_TAG="$TAG1" \
  python3 -u src/run_swegym_hf.py 2>&1 | tee "swegym_${TAG1}.log"
RUNDIR="results/swegym_${TAG1}_brief"

echo "== STAGE1b analysis $(date -Is)"
if [ "$STAGE" = "full" ]; then
  python3 -u src/analyze_swegym_tune.py --run "$RUNDIR" \
    --write-champion "$CHAMP" --profile-alpha "$SC_PROFILE_ALPHA" \
    2>&1 | tee "swegym_tune_analysis_${STAMP}.log"
else
  python3 -u src/analyze_swegym_tune.py --run "$RUNDIR" \
    2>&1 | tee "swegym_tune_analysis_${STAMP}.log"
  echo "SWEGYM_TUNE_SWEEP_DONE $(date -Is)"; exit 0
fi

# =====================================================================
# STAGE 2 (full mode): evaluate the SWE-Gym-tuned champion HELD-OUT.
# =====================================================================
if [ ! -f "$CHAMP" ]; then
  echo "WARN: no champion config was built (no positive TUNE regions?) -- skipping stage 2."
  echo "SWEGYM_TUNE_DONE $(date -Is)"; exit 0
fi
TAG2="champeval_${STAMP}"
echo "== STAGE2 held-out champion eval $(date -Is)  champion=$CHAMP -> results/swegym_${TAG2}_brief"
SC_HF_MODEL="$MODEL" SC_SWE_TAG="$TAG2" SC_CHAMPION_CONFIG="$CHAMP" \
  SC_E_ALPHAS="" SC_SWE_PLACEBO=0 \
  python3 -u src/run_swegym_hf.py 2>&1 | tee "swegym_${TAG2}.log"

echo "== FINAL analysis (held-out champion vs scalar) $(date -Is)"
python3 -u src/analyze_swegym_tune.py --run "$RUNDIR" \
  --champion-eval "results/swegym_${TAG2}_brief" \
  --profile-alpha "$SC_PROFILE_ALPHA" 2>&1 | tee "swegym_tune_final_${STAMP}.log"
echo "SWEGYM_TUNE_DONE $(date -Is)"
