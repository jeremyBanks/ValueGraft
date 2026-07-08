#!/usr/bin/env bash
# WAVE 2 — higher-risk architectures, run AFTER Wave 1 (may skip-with-reason).
# Gemma sliding-window/HybridCache (may be UNSUPPORTED — if it works, bonus 2nd
# dense/MoE de-confound), gpt-oss (sliding-window + attention sinks), GLM (trust_remote).
# 5 models across 4 pods; Gemma-4 dense/MoE pair split so each is observable.
set -uo pipefail
cd /Users/jeb/experimentation
declare -a ASSIGN=(
  "google/gemma-4-31B-it"          # gemma-4 dense (anchor-quality if it grafts)
  "google/gemma-4-26B-A4B-it"      # gemma-4 MoE   (anchor-quality if it grafts)
  "google/gemma-3-27b-it"          # gemma-3 sliding-window (major-version compare)
  "openai/gpt-oss-20b zai-org/GLM-4-32B-0414"
)
i=1
for MSET in "${ASSIGN[@]}"; do
  echo "== provisioning w2p${i}: $MSET"
  MODELS="$MSET" SC_POD_DISK=400 SC_POD_CLOUD=COMMUNITY \
    bash scripts/launch_pod.sh "w2p${i}" scripts/job_sweep.sh > "scratchpad/launch_w2p${i}.log" 2>&1 &
  i=$((i+1)); sleep 8
done
wait
echo "WAVE 2: attempted 4 pods (higher-risk). Expect some UNSUPPORTED (sliding-window)."
