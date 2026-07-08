#!/usr/bin/env bash
# SINGLE clean launch flow for the 11 text models (gated + observable). Reuses existing
# pods (w1/w2/w7) via their state files; provisions the other 8. Retries community-500s
# per model with backoff until it prints LAUNCHED or the deadline. No other launcher runs
# concurrently. Pre-flight token must be green (launch_pod.sh enforces).
set -uo pipefail
cd /Users/jeb/experimentation || exit 1
S=/private/tmp/claude-501/-Users-jeb-experimentation/bda7fb9f-f447-4890-904b-dde750ff3370/scratchpad
ENTRIES="1:Qwen/Qwen3-30B-A3B-Instruct-2507:24 2:Qwen/Qwen3-32B:24 3:Qwen/Qwen2.5-32B-Instruct:24 6:mistralai/Mixtral-8x7B-Instruct-v0.1:12 7:mistralai/Mistral-Small-24B-Instruct-2501:12 9:allenai/OLMo-2-0325-32B-Instruct:12 12:zai-org/GLM-4-32B-0414:12 13:openai/gpt-oss-20b:12 14:microsoft/phi-4:12 15:01-ai/Yi-1.5-34B-Chat:12 16:nvidia/Llama-3_3-Nemotron-Super-49B-v1_5:12"
DEADLINE=$(( $(date +%s) + 3600 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  pending=0
  for e in $ENTRIES; do
    i=${e%%:*}; rest=${e#*:}; M=${rest%:*}; C=${rest##*:}
    lg="$S/launch_w$i.log"
    # already launched (job running) this round?
    grep -q "LAUNCHED w$i at" "$lg" 2>/dev/null && continue
    pending=$((pending+1))
    echo "launching w$i ($M, $C convs)..."
    MODELS="$M" SC_CONV_LIMIT=$C SC_POD_DISK=400 bash scripts/launch_pod.sh "w$i" scripts/job_sweep.sh > "$lg" 2>&1
    grep -q "LAUNCHED w$i at" "$lg" 2>/dev/null && echo "  w$i LAUNCHED" || echo "  w$i failed this round ($(grep -aoE 'HTTP Error [0-9]+|REFUSING|not green' "$lg" | tail -1))"
    sleep 5
  done
  [ $pending -eq 0 ] && { echo "ALL 11 LAUNCHED"; break; }
  echo "-- round done; launched launched; retrying pending in 90s"
  sleep 90
done
