#!/usr/bin/env bash
# TRUSTED-APPARATUS positive control: gap_closure_cat.py (difflib alignment, SELF-GEN
# summary) on Qwen3-30B-A3B — the KNOWN-GOOD code that produced +0.156 referent. If
# referent returns positive, the effect + environment are sound and the cross_arch
# refactor's strict-alignment/think-strip divergence is the bug -> converge cross_arch
# onto this core. Runs detached; results in results/gap_closure_cat/<conv>.json.
set -uo pipefail
cd /workspace/exp 2>/dev/null || cd "$(dirname "$0")/.." || exit 1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "TRUSTED START $(date -Is)"; nvidia-smi || true
for f in /workspace/exp/.hf_key /workspace/exp/.huggingface_key; do
  [ -f "$f" ] && { export HF_TOKEN; HF_TOKEN="$(tr -d '[:space:]' < "$f")"; break; }
done
python3 -c "import torch,sys; print('cuda',torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)" || exit 1
python3 -m pip install -U "transformers>=4.57.0" accelerate safetensors huggingface_hub >/dev/null 2>&1 || true
export SC_HF_MODEL="Qwen/Qwen3-30B-A3B"
echo "== running trusted gap_closure_cat (self-gen, difflib) $(date -Is)"
python3 -u src/gap_closure_cat.py 2>&1 | tee gcc_trusted.log
echo "== TRUSTED SUMMARY: per-category effect across convs =="
python3 - <<'PY'
import json, glob, os
from collections import defaultdict
cat=defaultdict(list)
for f in glob.glob("results/gap_closure_cat/*.json"):
    if os.path.basename(f).startswith("_"): continue
    d=json.load(open(f))
    for p in (d.get("plants") or d.get("results") or []):
        c=p.get("category"); eb=p.get("raw_EB", p.get("e_minus_b"))
        if c is not None and eb is not None: cat[c].append(eb)
    # fallback: some layouts store per-category means directly
    for c,v in (d.get("by_category") or {}).items():
        m=v.get("raw_EB", v.get("mean")) if isinstance(v,dict) else None
        if m is not None: cat[c].append(m)
import statistics as st
for c in ["sense","referent","stance","ruled_out","evicted_fact"]:
    xs=cat.get(c,[])
    if xs: print(f"  {c}: mean_raw_EB={st.mean(xs):+.4f}  n={len(xs)}")
    else:  print(f"  {c}: (no data)")
PY
echo "TRUSTED DONE $(date -Is)"
