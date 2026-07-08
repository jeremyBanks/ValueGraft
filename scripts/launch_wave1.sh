#!/usr/bin/env bash
# WAVE 1 — low-risk, standard-arch models FIRST (banks the reliable sign-map +
# the Qwen dense/MoE de-confound). 11 models across 7 community pods: the 3 Qwen
# anchors run SOLO (heavy: placebo+alpha+champion), breadth paired. Each pod runs
# job_sweep.sh with its MODELS subset. Backgrounded; each pod is independent so a
# failed provision doesn't block the others. Re-run is idempotent per pod (state file).
# ONLY run this after: gate PASS + doubled corpus frozen.
set -uo pipefail
cd /Users/jeb/experimentation
declare -a ASSIGN=(
  "Qwen/Qwen3-30B-A3B"                                                      # anchor solo
  "Qwen/Qwen3-32B"                                                          # anchor solo
  "Qwen/Qwen2.5-32B-Instruct"                                              # anchor solo
  "mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mistral-Small-3.2-24B-Instruct-2506"
  "allenai/OLMo-2-0325-32B-Instruct 01-ai/Yi-1.5-34B-Chat"
  "Qwen/Qwen3.6-35B-A3B microsoft/phi-4"
  "Qwen/Qwen3.6-27B nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
)
i=1
for MSET in "${ASSIGN[@]}"; do
  echo "== provisioning w1p${i}: $MSET"
  MODELS="$MSET" SC_POD_DISK=400 SC_POD_CLOUD=COMMUNITY \
    bash scripts/launch_pod.sh "w1p${i}" scripts/job_sweep.sh > "scratchpad/launch_w1p${i}.log" 2>&1 &
  i=$((i+1)); sleep 8
done
wait
echo "WAVE 1: attempted 7 pods. Check scratchpad/launch_w1p*.log + scratchpad/pods.list"
