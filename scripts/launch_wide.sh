#!/usr/bin/env bash
# WIDE cross-arch sweep: per-token native render, 1 model/pod, 24 convs anchors / 12 breadth.
# Core sign-map (champion/placebo OFF; cheap follow-up later). Each pod independent + resilient.
set -uo pipefail
cd /Users/jeb/experimentation || exit 1
ANCHORS="Qwen/Qwen3-30B-A3B-Instruct-2507 Qwen/Qwen3-32B Qwen/Qwen2.5-32B-Instruct google/gemma-4-31B-it google/gemma-4-26B-A4B-it"
BREADTH="mistralai/Mixtral-8x7B-Instruct-v0.1 mistralai/Mistral-Small-24B-Instruct-2501 google/gemma-3-27b-it allenai/OLMo-2-0325-32B-Instruct Qwen/Qwen3.6-35B-A3B Qwen/Qwen3.6-27B zai-org/GLM-4-32B-0414 openai/gpt-oss-20b microsoft/phi-4 01-ai/Yi-1.5-34B-Chat nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
i=1
for M in $ANCHORS; do
  echo "== w$i ANCHOR (24 convs): $M"
  MODELS="$M" SC_CONV_LIMIT=24 SC_POD_DISK=400 SC_POD_CLOUD=COMMUNITY bash scripts/launch_pod.sh "w$i" scripts/job_sweep.sh > "$S/launch_w$i.log" 2>&1 &
  i=$((i+1)); sleep 8
done
for M in $BREADTH; do
  echo "== w$i BREADTH (12 convs): $M"
  MODELS="$M" SC_CONV_LIMIT=12 SC_POD_DISK=400 SC_POD_CLOUD=COMMUNITY bash scripts/launch_pod.sh "w$i" scripts/job_sweep.sh > "$S/launch_w$i.log" 2>&1 &
  i=$((i+1)); sleep 8
done
wait
echo "WIDE SWEEP: attempted 16 pods (1 model each). Check scratchpad/launch_w*.log + scratchpad/pods.list"
