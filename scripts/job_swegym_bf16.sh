#!/usr/bin/env bash
# SWE-GYM ANCHOR @ bf16 on the PRIMARY model -- SECONDARY / lower-effort.
# Strengthen the +0.0156 SWE-Gym anchor (E-tuned vs B, brief-summary condition) by
# replicating at higher N with a trajectory-bootstrap CI.
#
# The anchor condition = SC_SUMMARY=brief (the terse mechanism-isolation summary the
# +0.0156 was measured under), SC_E_ALPHA=0.75 (E-tuned's alpha). Decisive quantity:
# per-trajectory paired (E-tuned.tf_mean - B.tf_mean), bootstrapped over trajectories
# for a 95% CI. CI lower bound > 0 => the anchor replicates.
#
# NOTE on placebo: run_swegym_hf.py has NO source-corrupted placebo arm (unlike the
# chat cross_arch path). A true placebo would need a harness extension; this job's
# strengthening is the higher-N paired E-B CI. B is the honest within-trajectory
# control (same compacted context, summary swapped for the graft) -- report as such.
#
# PREREQS (coordinator): swegym.parquet must exist locally so launch_pod rsyncs it,
# and pyarrow must install on the pod (the job installs it). If swegym.parquet is
# absent locally, this job cannot run -- stage the parquet first.
#
# Env (forwarded / job defaults): SC_HF_MODEL, SC_SWE_TAG (30b_bf16), SC_SWE_N
# (default 150; prior was 75), SC_SUMMARY (brief = anchor), SC_E_ALPHA (0.75),
# SC_SHARD (0/1). bf16 loads on stock torch; only transformers>=4.57,<5 pinned.
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${SC_HF_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
TAG="${SC_SWE_TAG:-30b_bf16}"
export SC_SWE_N="${SC_SWE_N:-150}"
export SC_SUMMARY="${SC_SUMMARY:-brief}"
export SC_E_ALPHA="${SC_E_ALPHA:-0.75}"
export SC_SHARD="${SC_SHARD:-0/1}"
SUMM_TAG="brief"; [ "$SC_SUMMARY" = "brief" ] || SUMM_TAG="prod"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "START swegym-bf16 $(date -Is)  model=$MODEL N=$SC_SWE_N summary=$SC_SUMMARY alpha=$SC_E_ALPHA shard=$SC_SHARD"
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
[ -f swegym.parquet ] || { echo "FATAL: swegym.parquet missing on pod (stage it locally so launch_pod rsyncs it)"; exit 4; }

echo "== run swegym eval $(date -Is)"
SC_HF_MODEL="$MODEL" SC_SWE_TAG="$TAG" python3 -u src/run_swegym_hf.py 2>&1 | tee "swegym_${TAG}_${SUMM_TAG}.log"

echo "== SUMMARY: paired (arm - B) trajectory bootstrap CI, per grafted arm $(date -Is)"
python3 - "$TAG" "$SUMM_TAG" <<'PY'
import json, glob, random, sys
tag, summ = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"results/swegym_{tag}_{summ}/t*.json"))
# collect per-arm paired diffs (arm.tf_mean - B.tf_mean) for every grafted arm
# present; E-champion appears only when SC_CHAMPION_CONFIG was set.
by_arm = {}
champ_label = None
for f in files:
    try:
        d = json.load(open(f)); arms = d["arms"]; b = arms["B"]["tf_mean"]
    except Exception:
        continue
    for name in ("E-tuned", "E-champion"):
        if name in arms and arms[name].get("tf_mean") is not None:
            by_arm.setdefault(name, []).append(arms[name]["tf_mean"] - b)
            if name == "E-champion":
                iv = arms[name].get("intervention") or {}
                champ_label = (iv.get("champion") or {}).get("label") or champ_label
def boot(diffs):
    n = len(diffs)
    if n == 0:
        return None
    mean = sum(diffs) / n
    random.seed(0)
    bs = []
    for _ in range(10000):
        bs.append(sum(diffs[random.randrange(n)] for _ in range(n)) / n)
    bs.sort()
    return n, mean, bs[249], bs[9749], sum(1 for x in diffs if x > 0)
for name in ("E-tuned", "E-champion"):
    r = boot(by_arm.get(name, []))
    if r is None:
        print(f"  {name}: (not present)"); continue
    n, mean, lo, hi, pos = r
    flag = "CI>0" if lo > 0 else "CI spans 0"
    extra = f"  [champion={champ_label}]" if name == "E-champion" else ""
    print(f"  {name} - B  mean={mean:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]  "
          f"pct_helped={pos}/{n}  -> {flag}{extra}")
# head-to-head champion vs scalar where both present (paired over trajectories)
if by_arm.get("E-tuned") and by_arm.get("E-champion") and \
   len(by_arm["E-tuned"]) == len(by_arm["E-champion"]):
    d = [c - s for s, c in zip(by_arm["E-tuned"], by_arm["E-champion"])]
    r = boot(d)
    if r:
        n, mean, lo, hi, pos = r
        flag = "champion BEATS scalar (CI>0)" if lo > 0 else "not distinguishable (CI spans 0)"
        print(f"  E-champion - E-tuned (paired)  mean={mean:+.4f}  "
              f"95%CI=[{lo:+.4f}, {hi:+.4f}]  -> {flag}")
print(f"  (prior scalar anchor +0.0156 @ N=75, BRIEF)")
PY
echo "SWEGYM_BF16_DONE $(date -Is)"
