#!/usr/bin/env bash
# Resilient retry: keep trying to provision the wide-sweep pods that failed the community 500s,
# a few at a time with backoff, until all up or the deadline. Each success launches its job.
set -uo pipefail
cd /Users/jeb/experimentation
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
# w-index : model : conv_limit   (w1 already reused)
ENTRIES=(
 "2:Qwen/Qwen3-32B:24" "3:Qwen/Qwen2.5-32B-Instruct:24" "4:google/gemma-4-31B-it:24"
 "5:google/gemma-4-26B-A4B-it:24" "6:mistralai/Mixtral-8x7B-Instruct-v0.1:12"
 "7:mistralai/Mistral-Small-3.2-24B-Instruct-2506:12" "8:google/gemma-3-27b-it:12"
 "9:allenai/OLMo-2-0325-32B-Instruct:12" "10:Qwen/Qwen3.6-35B-A3B:12" "11:Qwen/Qwen3.6-27B:12"
 "12:zai-org/GLM-4-32B-0414:12" "13:openai/gpt-oss-20b:12" "14:microsoft/phi-4:12"
 "15:01-ai/Yi-1.5-34B-Chat:12" "16:nvidia/Llama-3_3-Nemotron-Super-49B-v1_5:12"
)
DEADLINE=$(( $(date +%s) + 3000 ))   # ~50 min of retrying
ROUND=0
while [ $(date +%s) -lt $DEADLINE ]; do
  ROUND=$((ROUND+1)); LAUNCHED=0; MISSING=0
  for e in "${ENTRIES[@]}"; do
    i=${e%%:*}; rest=${e#*:}; M=${rest%:*}; C=${rest##*:}
    # already provisioned? (state file with an id)
    if [ -f ".pod_w${i}_state.json" ] && grep -q '"id"' ".pod_w${i}_state.json" 2>/dev/null; then continue; fi
    MISSING=$((MISSING+1))
    MODELS="$M" SC_CONV_LIMIT=$C SC_POD_DISK=400 SC_POD_CLOUD=COMMUNITY bash scripts/launch_pod.sh "w$i" scripts/job_sweep.sh > "$S/launch_w$i.log" 2>&1 &
    LAUNCHED=$((LAUNCHED+1)); sleep 12
  done
  wait
  UP=$(ls .pod_w*_state.json 2>/dev/null | wc -l | tr -d ' ')
  echo "RETRY round $ROUND: $UP pods provisioned, $MISSING still missing $(date -Is)"
  [ "$MISSING" -eq 0 ] && { echo "ALL PODS UP"; break; }
  sleep 150
done
